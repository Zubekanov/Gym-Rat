import functools

import gymnasium as gym
import numpy as np
import random
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper
from pettingzoo.utils.wrappers.capture_stdout import CaptureStdoutWrapper

CLM_REWARD = 0.001
TOK_REWARD = 0.00075
WIN_REWARD = 1.0
IMBECILE_PENALTY = 10.0

GRID_SIZE = 9
TURN_LIMIT = 41

# Bitmasks
no_claim    = 0b0000
p0_token    = 0b0001
p0_claim    = 0b0010
p1_token    = 0b0100
p1_claim    = 0b1000

grid_values = {
	0: {"token": p0_token, "claim": p0_claim},
	1: {"token": p1_token, "claim": p1_claim},
}

class PopuGameEnv(AECEnv):
	metadata = {"render_modes": ["human"], "name": "popugame_v0"}
	render_mode = "human"

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.possible_agents = ["player_0", "player_1"]
		self.agent_name_mapping = {"player_0": 0, "player_1": 1}

		self.grid_size = GRID_SIZE
		self.turn_limit = TURN_LIMIT

		# will be set in reset()
		self.grid = None
		self.turn = None
		self.legal_moves = None
		self.prev_score = [0, 0]
		self.curr_score = [0, 0]

		# cache spaces
		self._observation_spaces = {a: Box(0, 15, (self.grid_size, self.grid_size), np.uint8)
									for a in self.possible_agents}
		self._action_spaces = {a: Discrete(self.grid_size * self.grid_size)
							   for a in self.possible_agents}
		
		self.reset()

	def observation_space(self, agent):
		return self._observation_spaces[agent]

	def action_space(self, agent):
		return self._action_spaces[agent]

	def reset(self, seed=None, options=None):
		# seed & super-init
		if seed is not None:
			self.np_random, _ = seeding.np_random(seed)
		self.agents = self.possible_agents[:]
		# initialize empties
		self.rewards = {agent: 0.0 for agent in self.agents}
		self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
		self.terminations = {agent: False for agent in self.agents}
		self.truncations = {agent: False for agent in self.agents}
		self.infos = {agent: {} for agent in self.agents}

		self.prev_score = [0, 0]
		self.curr_score = [0, 0]

		# game state
		self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
		self.turn = 0
		self.legal_moves = {
			"player_0": np.ones((self.grid_size, self.grid_size), dtype=bool),
			"player_1": np.ones((self.grid_size, self.grid_size), dtype=bool),
		}

		for agent in self.agents:
			self.infos[agent]["action_mask"] = self.legal_moves[agent].flatten().astype(np.int8)

		# set up agent iterator
		self._agent_selector = AgentSelector(self.agents)
		self.agent_selection = self._agent_selector.next()

		# **clear any residual rewards** (recommended by API)
		self._clear_rewards()

		# initial observations
		self.observations = {a: self.observe(a) for a in self.agents}
		return self.observations, self.infos

	def observe(self, agent):
		# here we return the raw 2D grid; agents can flatten if needed
		return self.grid.copy()

	def step(self, action):
		agent = self.agent_selection

		# skip if already done
		if self.terminations[agent] or self.truncations[agent]:
			self._was_dead_step(action)
			return
		
		if not isinstance(action, int):
			action = int(action)

		# clear out old instantaneous rewards
		self._cumulative_rewards[agent] = 0.0
		self._clear_rewards()

		# decode & apply the move
		row, col = divmod(action, self.grid_size)
		if self.legal_moves[agent][row, col]:
			n_player = self.agent_name_mapping[agent]
			claim_delta = self._check_claim(n_player, row, col)

			# REWARD CALCULATION SECTION
			# New rewards based on claim differences.
			# Agents still durdle in the corner, so assigning points for tokens as well.

			p0_claims = int(np.sum((self.grid & p0_claim) != 0))
			p1_claims = int(np.sum((self.grid & p1_claim) != 0))
			p0_tokens = int(np.sum((self.grid & p0_token) != 0))
			p1_tokens = int(np.sum((self.grid & p1_token) != 0))

			p0_score = p0_claims * CLM_REWARD + p0_tokens * TOK_REWARD
			p1_score = p1_claims * CLM_REWARD + p1_tokens * TOK_REWARD

			self.prev_score = self.curr_score
			self.curr_score = [p0_score, p1_score]

			score_delta = self.curr_score[n_player] - self.prev_score[n_player]
			# If score decreased and no claims were made, it was a strictly bad move.
			# Dock massive points for being an inbecile.
			if score_delta < 0 and claim_delta == 0:
				score_delta -= IMBECILE_PENALTY

			self.rewards[agent] = score_delta

			self.turn += 1
			occupied = (self.grid & (p0_token | p1_token)) != 0
			c0 = (self.grid & p0_claim) != 0
			c1 = (self.grid & p1_claim) != 0
			self.legal_moves["player_0"] = ~occupied & ~c1
			self.legal_moves["player_1"] = ~occupied & ~c0

			if self.turn >= self.turn_limit:
				score0 = int(np.sum((self.grid & p0_claim) != 0))
				score1 = int(np.sum((self.grid & p1_claim) != 0))
				if score0 > score1:
					self.rewards["player_0"] += WIN_REWARD
				elif score1 > score0:
					self.rewards["player_1"] += WIN_REWARD
				for a in self.agents:
					self.terminations[a] = True
		
		# accumulate into the internal cumulative totals
		if not any (self.terminations.values()):
			self.agent_selection = self._agent_selector.next()
		self._accumulate_rewards()

		# refresh observations & infos
		self.observations = {a: self.observe(a) for a in self.agents}
		for agent in self.agents:
			# update infos
			self.infos[agent]["action_mask"] = self.legal_moves[agent].flatten().astype(np.int8)

	def render(self):
		print(self.grid)

	def close(self):
		pass

	# --- internal claim logic ---

	def _check_claim(self, player, row: int, col: int) -> int:
		token = grid_values[player]["token"]
		self.grid[row, col] |= token

		mark_for_claim = np.zeros_like(self.grid, dtype=bool)
		mark_for_remove = np.zeros_like(self.grid, dtype=bool)

		def proc(start, end, step):
			cont = self._check_line(token, start, end, step)
			if cont["continuous"] >= 3:
				# mark removals
				r0, c0 = cont["start"]
				for i in range(cont["continuous"]):
					mark_for_remove[r0 + i*step[0], c0 + i*step[1]] = True
				# expand claims
				self.modify_claims(player, mark_for_claim, cont["start"], step)

		# horizontal
		proc((row, max(0, col-2)), (row, min(self.grid_size-1, col+2)), (0, 1))
		# vertical
		proc((max(0, row-2), col), (min(self.grid_size-1, row+2), col), (1, 0))
		# diag TL-BR
		proc((row-2, col-2), (row+2, col+2), (1, 1))
		# diag TR-BL
		proc((row-2, col+2), (row+2, col-2), (1, -1))

		prev_claims = int(np.sum((self.grid & grid_values[player]["claim"]) != 0))

		# apply removals & claims
		self.grid[mark_for_remove] = no_claim
		opp_claim = grid_values[1-player]["claim"]
		self.grid[mark_for_claim] &= (0b1111 - opp_claim)
		self.grid[mark_for_claim] |= grid_values[player]["claim"]

		curr_claims = int(np.sum((self.grid & grid_values[player]["claim"]) != 0))

		return curr_claims - prev_claims

	def modify_claims(self, player, mark_for_claim, start, step):
		r, c = start
		# forward
		i = 0
		while True:
			r0, c0 = r + i*step[0], c + i*step[1]
			if self.out_of_bounds(r0, c0) or (self.grid[r0, c0] & grid_values[1-player]["token"]):
				break
			mark_for_claim[r0, c0] = True
			i += 1
		# backward
		i = -1
		while True:
			r0, c0 = r + i*step[0], c + i*step[1]
			if self.out_of_bounds(r0, c0) or (self.grid[r0, c0] & grid_values[1-player]["token"]):
				break
			mark_for_claim[r0, c0] = True
			i -= 1

	def out_of_bounds(self, row: int, col: int) -> bool:
		return not (0 <= row < self.grid_size and 0 <= col < self.grid_size)

	def _check_line(self, mask, start, end, step):
		max_cont, curr_cont = 0, 0
		max_start = max_end = None
		curr_start = None
		r, c = start
		while True:
			if self.out_of_bounds(r, c):
				break
			hit = bool(self.grid[r, c] & mask)
			if hit:
				if curr_cont == 0:
					curr_start = (r, c)
				curr_cont += 1
			else:
				if curr_cont > max_cont:
					max_cont = curr_cont
					max_start = curr_start
					max_end = (r - step[0], c - step[1])
				curr_cont = 0
			r += step[0]; c += step[1]
			if (step[0] and ((step[0] > 0 and r > end[0]) or (step[0] < 0 and r < end[0]))) or \
			   (step[1] and ((step[1] > 0 and c > end[1]) or (step[1] < 0 and c < end[1]))):
				break
		# final check
		if curr_cont > max_cont:
			max_cont = curr_cont
			max_start = curr_start
			max_end = (r - step[0], c - step[1])
		return {"start": max_start, "end": max_end, "continuous": max_cont}
	
	@staticmethod
	def env(**kwargs):
		base = PopuGameEnv(**kwargs)
		env  = OrderEnforcingWrapper(base)
		return env

class GreedyAgent:
    def __init__(self, env, agent_name, claim_weight=100):
        self.env          = env
        self.agent_name   = agent_name
        self.pidx         = env.agent_name_mapping[agent_name]
        self.N            = env.grid_size
        self.claim_weight = claim_weight

        # precompute center‐distance for tie-breaking
        center = (self.N - 1) / 2
        self.center_d = {
            (r, c): abs(r - center) + abs(c - center)
            for r in range(self.N) for c in range(self.N)
        }

        # bitmasks for your token & your claim
        self.token_bit = [0b0001, 0b0100][self.pidx]
        self.claim_bit = [0b0010, 0b1000][self.pidx]

    def act(self, grid, mask_flat):
        mask2 = mask_flat.astype(bool).reshape(self.N, self.N)

        best_score = -1e9
        best_moves = []

        neigh_offsets = [
            (dr, dc)
            for dr in range(-2, 3)
            for dc in range(-2, 3)
            if abs(dr) + abs(dc) <= 2 and not (dr == 0 and dc == 0)
        ]

        real_grid = self.env.grid  # so we can restore it

        for r in range(self.N):
            for c in range(self.N):
                if not mask2[r, c]:
                    continue

                already_claimed = bool(grid[r, c] & self.claim_bit)

                if already_claimed:
                    # no heuristic credit on a square you already claimed
                    max_run     = 0
                    neigh_count = 0
                    center_pref = 0
                else:
                    max_run = 1
                    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                        cnt = 1
                        # forward
                        for k in (1,2):
                            rr, cc = r + dr*k, c + dc*k
                            if (0 <= rr < self.N and 0 <= cc < self.N
                                and (grid[rr,cc] & self.token_bit)):
                                cnt += 1
                            else:
                                break
                        # backward
                        for k in (1,2):
                            rr, cc = r - dr*k, c - dc*k
                            if (0 <= rr < self.N and 0 <= cc < self.N
                                and (grid[rr,cc] & self.token_bit)):
                                cnt += 1
                            else:
                                break
                        max_run = max(max_run, cnt)

                    neigh_count = 0
                    for dr, dc in neigh_offsets:
                        rr, cc = r + dr, c + dc
                        if (0 <= rr < self.N and 0 <= cc < self.N
                            and (grid[rr,cc] & self.token_bit)):
                            neigh_count += 1

                    center_pref = -self.center_d[(r, c)]

                temp = grid.copy()
                try:
                    self.env.grid = temp
                    claim_delta = self.env._check_claim(self.pidx, r, c)
                finally:
                    self.env.grid = real_grid

                score = (
                    (1000 if max_run >= 3 else 0)
                    + max_run * 10
                    + neigh_count * 2
                    + center_pref
                    + self.claim_weight * claim_delta
                )

                if score > best_score:
                    best_score, best_moves = score, [(r, c)]
                elif score == best_score:
                    best_moves.append((r, c))

        if best_moves:
            r, c = random.choice(best_moves)
            return r * self.N + c

        # fallback (shouldn’t happen if there’s any legal move)
        return random.choice(np.where(mask_flat)[0])


class BlockerAgent:
	def __init__(self, env, agent_name):
		from copy import deepcopy
		self.env        = env
		self.agent_name = agent_name
		self.pidx        = env.agent_name_mapping[agent_name]
		self.N           = env.grid_size
		other            = [a for a in env.possible_agents if a != agent_name][0]
		self.opp_pidx    = env.agent_name_mapping[other]
		self.greedy      = GreedyAgent(env, agent_name)

	def act(self, grid, mask_flat):
		mask2 = mask_flat.astype(bool).reshape(self.N, self.N)
		opp_token = [0b0001, 0b0100][self.opp_pidx]

		threats = []
		for idx, ok in enumerate(mask_flat):
			if not ok: continue
			r, c = divmod(idx, self.N)
			run = 1
			for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
				cnt = 1
				for k in (1,2):
					rr, cc = r + dr*k, c + dc*k
					if 0 <= rr < self.N and 0 <= cc < self.N and (grid[rr,cc] & opp_token):
						cnt += 1
					else:
						break
				for k in (1,2):
					rr, cc = r - dr*k, c - dc*k
					if 0 <= rr < self.N and 0 <= cc < self.N and (grid[rr,cc] & opp_token):
						cnt += 1
					else:
						break
				run = max(run, cnt)
			threats.append(((r,c), run))

		if not threats:
			# no opponent moves? fallback
			return self.greedy.act(grid, mask_flat)

		# sort threats descending by run
		threats.sort(key=lambda x: -x[1])
		top_run = threats[0][1]

		top_cells = [pos for pos, run in threats if run == top_run]
		for (r,c) in top_cells:
			if mask2[r,c]:
				return r * self.N + c

		best_block, best_reduction = None, 0
		original_max = top_run
		for pos, run in threats[1:4]:  # look at next three
			r, c = pos
			if not mask2[r,c]:
				continue
			# simulate blocking: mark it as occupied so opp can't play there
			# then recompute their max run over remaining legal
			sim_max = 0
			for pos2, _ in threats:
				if pos2 == pos: continue
				rr, cc = pos2
				# compute run at pos2 ignoring any token at blocked r,c
				cnt_run = 1
				for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
					cnt = 1
					for k in (1,2):
						rrr, ccc = rr + dr*k, cc + dc*k
						if (0 <= rrr < self.N and 0 <= ccc < self.N
								and (grid[rrr,ccc] & opp_token)
								and not (rrr == r and ccc == c)):
							cnt += 1
						else:
							break
					for k in (1,2):
						rrr, ccc = rr - dr*k, cc - dc*k
						if (0 <= rrr < self.N and 0 <= ccc < self.N
								and (grid[rrr,ccc] & opp_token)
								and not (rrr == r and ccc == c)):
							cnt += 1
						else:
							break
					cnt_run = max(cnt_run, cnt)
				sim_max = max(sim_max, cnt_run)
			reduction = original_max - sim_max
			if reduction > best_reduction:
				best_reduction = reduction
				best_block     = (r, c)

		if best_block:
			r, c = best_block
			return r * self.N + c

		return self.greedy.act(grid, mask_flat)