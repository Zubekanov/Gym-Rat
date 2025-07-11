import csv
import os
import random
import re
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from environment.popugame_env import PopuGameEnv, GreedyAgent, BlockerAgent

# === Hyperparameters ===
LR               = 1e-4
GAMMA            = 0.99
DEFAULT_EPISODES = 50000
SAVE_EVERY       = 1000
BATCH_SIZE       = 10       # number of episodes per update
BETA_START       = 0.1
BETA_END         = 0.01
EVAL_GAMES       = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CLI args ===
parser = argparse.ArgumentParser()
parser.add_argument("--resume-dir", type=str, default=None,
					help="Directory with checkpoints to resume from")
parser.add_argument("--episodes",  type=int, default=DEFAULT_EPISODES,
					help="Total number of episodes to train")
args = parser.parse_args()

# === Determine MODEL_PATH and start episode ===
if args.resume_dir:
	MODEL_PATH = args.resume_dir
	ckpt_files = [f for f in os.listdir(MODEL_PATH) if re.match(r".*_ep\d+\.pt$", f)]
	if not ckpt_files:
		raise ValueError(f"No checkpoints found in {MODEL_PATH}")
	last_ep  = max(int(re.search(r"_ep(\d+)\.pt$", f).group(1)) for f in ckpt_files)
	start_ep = last_ep + 1
	print(f"Resuming from episode {last_ep}. Starting at {start_ep}.")
else:
	ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
	MODEL_PATH = os.path.join("trained_agents", ts)
	start_ep   = 1

os.makedirs(MODEL_PATH, exist_ok=True)
stats_csv = os.path.join(MODEL_PATH, "curriculum_stats.csv")
if not os.path.isfile(stats_csv):
    with open(stats_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "opponent_type", "count", "avg_score", "avg_margin", "win_pct"])
dir_resume = bool(args.resume_dir)

# === Environment setup ===
env       = PopuGameEnv()
agents    = env.possible_agents
env.reset()
obs_size  = np.prod(env.observe(agents[0]).shape)
act_size  = env.action_space(agents[0]).n

# === Policy network ===
class PolicyNet(nn.Module):
	def __init__(self, obs_size, action_size):
		super().__init__()
		self.model = nn.Sequential(
			nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
			nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
			nn.Flatten(),
			nn.Linear(64 * 9 * 9, 256), nn.ReLU(),
			nn.Linear(256, action_size)
		)

	def forward(self, x):
		return self.model(x)

# === Init policies, optimizers, schedulers ===
policies   = {a: PolicyNet(obs_size, act_size).to(DEVICE) for a in agents}
optimizers = {a: optim.Adam(policies[a].parameters(), lr=LR)      for a in agents}
schedulers = {a: StepLR(optimizers[a], step_size=100, gamma=0.995) for a in agents}

if dir_resume:
	for a in agents:
		ckpt = os.path.join(MODEL_PATH, f"{a}_ep{last_ep}.pt")
		policies[a].load_state_dict(torch.load(ckpt, map_location=DEVICE))
	print("Loaded pretrained weights.")

# === Buffers for batch updates and histories ===
batch_logps     = {a: [] for a in agents}
batch_returns   = {a: [] for a in agents}
batch_entropies = {a: [] for a in agents}
batch_count     = 0

reward_history  = {a: deque(maxlen=100) for a in agents}
entropy_history = {a: deque(maxlen=100) for a in agents}

# === Stats for curriculum opponents per 100-episode window ===
opp_stats = {
	"latest":     {"count": 0, "sum_score": 0.0, "sum_margin": 0.0, "wins": 0},
	"historical": {"count": 0, "sum_score": 0.0, "sum_margin": 0.0, "wins": 0},
	"greedy":     {"count": 0, "sum_score": 0.0, "sum_margin": 0.0, "wins": 0},
	"blocker":    {"count": 0, "sum_score": 0.0, "sum_margin": 0.0, "wins": 0},
}

# === Helper: sample an opponent for the curriculum ===
def sample_opponent(agent_name):
	pool = ["latest", "historical", "greedy", "blocker"]
	weights = [0.4, 0.3, 0.15, 0.15]
	choice = random.choices(pool, weights)[0]

	if choice == "latest":
		return choice, ("policy", policies[agent_name])

	if choice == "historical":
		files = [f for f in os.listdir(MODEL_PATH) if f.startswith(agent_name)]
		eps   = sorted(int(re.search(r"_ep(\d+)\.pt", f).group(1)) for f in files)
		if len(eps) <= 1:
			return "latest", ("policy", policies[agent_name])
		ep = random.choice(eps[:-1])
		m  = PolicyNet(obs_size, act_size).to(DEVICE)
		m.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{agent_name}_ep{ep}.pt"),
									 map_location=DEVICE))
		m.eval()
		return choice, ("policy", m)

	if choice == "greedy":
		return choice, ("greedy", GreedyAgent(env, agent_name))

	# blocker
	return choice, ("blocker", BlockerAgent(env, agent_name))

# === Helper: evaluate against all old checkpoints ===
def evaluate_against_old_snapshots(latest_agent, own_policy, opponent, ckpt_dir, old_eps):
	results = {}
	for ep in sorted(old_eps):
		opp_model = PolicyNet(obs_size, act_size).to(DEVICE)
		opp_model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{opponent}_ep{ep}.pt"),
											 map_location=DEVICE))
		opp_model.eval()

		total_r = 0.0
		for _ in range(EVAL_GAMES):
			env.reset()
			game_rew = {a: 0.0 for a in agents}
			for ag in env.agent_iter():
				obs, rew, term, trunc, info = env.last()
				done = term or trunc
				if not done:
					obs_t = torch.stack(
						[(torch.tensor(obs) & m) != 0
						 for m in [0b0001, 0b0010, 0b0100, 0b1000]]
					).float().unsqueeze(0).to(DEVICE)
					mask  = torch.tensor(info["action_mask"], dtype=torch.bool, device=DEVICE)
					if ag == latest_agent:
						logits = own_policy(obs_t).squeeze(0)
					else:
						logits = opp_model(obs_t).squeeze(0)
					logits = logits.masked_fill(~mask, float("-inf"))
					dist   = torch.distributions.Categorical(logits=logits)
					action = dist.sample().item()
				else:
					action = None
				env.step(action)
				game_rew[ag] += rew
			total_r += game_rew[latest_agent]
		results[ep] = total_r / EVAL_GAMES
	return results

# === Training loop ===
dot_count = 0
for episode in range(start_ep, args.episodes + 1):
	# alternate which agent is being trained
	train_agent = agents[episode % 2]
	opp_agent   = agents[(episode + 1) % 2]
	opp_type, (opp_kind, opp_obj) = sample_opponent(opp_agent)

	env.reset()
	ep_logps, ep_rews, ep_ents = {a: [] for a in agents}, {a: [] for a in agents}, {a: [] for a in agents}

	# run one episode
	for agent in env.agent_iter():
		obs, rew, term, trunc, info = env.last()
		done = term or trunc

		if not done and agent == train_agent:
			# --- training agent turn ---
			obs_t = torch.tensor(obs, dtype=torch.int64, device=DEVICE)
			chans = [(obs_t & m) != 0 for m in [1,2,4,8]]
			x     = torch.stack(chans,0).float().unsqueeze(0)
			mask  = torch.tensor(info["action_mask"], dtype=torch.bool, device=DEVICE)
			logits= policies[agent](x).squeeze(0)
			logits= logits.masked_fill(~mask, float("-inf"))
			dist  = torch.distributions.Categorical(logits=logits)
			action= dist.sample()
			ep_logps[agent].append(dist.log_prob(action))
			ep_ents[agent].append(dist.entropy())
			a_out = action.item()

		elif not done and agent == opp_agent:
			# --- opponent turn ---
			if opp_kind == "policy":
				obs_t = torch.tensor(obs, dtype=torch.int64, device=DEVICE)
				chans = [(obs_t & m) != 0 for m in [1,2,4,8]]
				x     = torch.stack(chans,0).float().unsqueeze(0)
				mask  = torch.tensor(info["action_mask"], dtype=torch.bool, device=DEVICE)
				logits= opp_obj(x).squeeze(0).masked_fill(~mask, float("-inf"))
				dist  = torch.distributions.Categorical(logits=logits)
				a_out = dist.sample().item()
			else:
				a_out = opp_obj.act(obs, info["action_mask"])
		else:
			a_out = None

		env.step(a_out)
		ep_rews[agent].append(rew)

	# update reward & entropy histories
	for a in agents:
		total = sum(ep_rews[a])
		reward_history[a].append(total)
		ep_ent_mean = torch.stack(ep_ents[a]).mean().item() if ep_ents[a] else 0.0
		entropy_history[a].append(ep_ent_mean)

	# append to batch buffers for train_agent
	R, returns = 0, []
	for r in reversed(ep_rews[train_agent]):
		R = r + GAMMA * R
		returns.insert(0, R)
	ret_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
	ret_tensor = (ret_tensor - ret_tensor.mean()) / (ret_tensor.std() + 1e-8)
	batch_logps[train_agent].extend(ep_logps[train_agent])
	batch_returns[train_agent].extend(ret_tensor)
	batch_entropies[train_agent].append(torch.stack(ep_ents[train_agent]).mean())

	batch_count += 1

	total_train = sum(ep_rews[train_agent])
	total_opp   = sum(ep_rews[opp_agent])
	margin      = total_train - total_opp

	st = opp_stats[opp_type]
	st["count"]      += 1
	st["sum_score"]  += total_train
	st["sum_margin"] += margin
	st["wins"]       += int(margin > 0)
	
	# perform update every BATCH_SIZE episodes
	if batch_count >= BATCH_SIZE:
		frac = (episode - 1) / args.episodes
		beta = BETA_START - frac * (BETA_START - BETA_END)

		optimizers[train_agent].zero_grad()
		loss = 0
		for lp, R in zip(batch_logps[train_agent], batch_returns[train_agent]):
			loss -= lp * R
		ent_mean = torch.stack(batch_entropies[train_agent]).mean()
		loss   -= beta * ent_mean

		loss.backward()
		torch.nn.utils.clip_grad_norm_(policies[train_agent].parameters(), 0.5)
		optimizers[train_agent].step()
		schedulers[train_agent].step()

		# clear buffers for next batch
		batch_logps[train_agent]     = []
		batch_returns[train_agent]   = []
		batch_entropies[train_agent] = []
		batch_count = 0

	# inline progress dots
	if episode % 10 == 0:
		print('.', end='', flush=True)
		dot_count += 1

	# periodic saving & evaluation
	if episode % SAVE_EVERY == 0 or episode == args.episodes:
		if dot_count:
			print('\r' + ' ' * dot_count + '\r', end='')
			dot_count = 0

		# save snapshots
		for a in agents:
			torch.save(policies[a].state_dict(),
					   os.path.join(MODEL_PATH, f"{a}_ep{episode}.pt"))
		print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved models at episode {episode}")

		# evaluate new snapshots vs all older
		ckpts = os.listdir(MODEL_PATH)
		eps_by_agent = {
			a: sorted(int(re.search(rf"{a}_ep(\d+)\.pt", f).group(1))
					  for f in ckpts if f.startswith(a))
			for a in agents
		}
		for ag in agents:
			latest_ep = eps_by_agent[ag][-1]
			own_pol   = policies[ag]
			opp       = [o for o in agents if o != ag][0]
			old_eps   = eps_by_agent[opp][:-1]
			if not old_eps:
				continue
			results = evaluate_against_old_snapshots(
				latest_agent=ag,
				own_policy=own_pol,
				opponent=opp,
				ckpt_dir=MODEL_PATH,
				old_eps=old_eps
			)
			scores = ",\n\t".join(f"vs {opp}_ep{e}: {r:.2f}" for e,r in results.items())
			print(f"[Eval@ep{latest_ep}] {ag} average reward:\n\t{scores}")

	# every 100 episodes: print RL metrics + curriculum stats
	if episode % 100 == 0:
		if dot_count:
			print('\r' + ' ' * dot_count + '\r', end='')
			dot_count = 0

		avg_str = " | ".join(
			f"{a}:[R={np.mean(reward_history[a]):.2f} E={np.mean(entropy_history[a]):.2f}]"
			for a in agents
		)
		print(f"\nEpisode {episode:5d} | {avg_str}")
		print("  vs-opponent-type stats (last 100 eps):")
		print(" Opponent Type | Cnt | AvgScore | AvgMargin | Win %")
		print("---------------|-----|----------|-----------|-------")
		for typ, s in opp_stats.items():
			cnt = s["count"]
			if cnt == 0:
				print(f"    {typ:>10s} |   0 |     --   |      --   |    --")
			else:
				avg_sc   = s["sum_score"]  / cnt
				avg_marg = s["sum_margin"] / cnt
				winpct   = 100.0 * s["wins"]   / cnt
				print(f"    {typ:>10s} | {cnt:3d} | {avg_sc:8.3f} | {avg_marg:9.3f} | {winpct:5.1f}")

		with open(stats_csv, "a", newline="") as f:
			writer = csv.writer(f)
			for typ, s in opp_stats.items():
				cnt = s["count"]
				if cnt > 0:
					avg_sc   = s["sum_score"]  / cnt
					avg_marg = s["sum_margin"] / cnt
					winpct   = 100.0 * s["wins"]   / cnt
				else:
					avg_sc = avg_marg = winpct = 0.0
				writer.writerow([episode, typ, cnt, f"{avg_sc:.3f}", f"{avg_marg:.3f}", f"{winpct:.1f}"])

		# reset stats
		for s in opp_stats.values():
			s["count"]      = 0
			s["sum_score"]  = 0.0
			s["sum_margin"] = 0.0
			s["wins"]       = 0
