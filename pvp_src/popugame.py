import tkinter as tk
from typing import Tuple
import numpy as np

_DEFAULT_SIZE = 9
_DEFAULT_TURN_LIMIT = 40

no_claim = 0b0000

p0_token = 0b0001
p0_claim = 0b0010

p1_token = 0b0100
p1_claim = 0b1000

grid_values = {
	0: {"token": p0_token, "claim": p0_claim},
	1: {"token": p1_token, "claim": p1_claim},
}

# ANSI escape codes for colors
RESET = '\x1b[0m'
FG_GREEN = '\x1b[32m'
FG_BLUE = '\x1b[34m'
BG_GREEN = '\x1b[42m'
BG_BLUE = '\x1b[44m'

class PopuGameGUI:
	def __init__(self, size=_DEFAULT_SIZE, turn_limit=_DEFAULT_TURN_LIMIT):
		print("[DEBUG] Initializing PopuGameGUI")
		self.grid_size = size
		self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
		self.turn_limit = turn_limit
		self.turn = 0

		self.window = tk.Tk()
		self.window.title("PopuGame")
		self.buttons = [[None for _ in range(size)] for _ in range(size)]

		self.player_selector = 0
		self.legal_moves = None

		self.scores = [0, 0]

		self.create_widgets()
		print("[DEBUG] Widgets created, now resetting game")
		self.reset_game()
		self.window.mainloop()

	def create_widgets(self):
		print("[DEBUG] Creating score and turn labels")
		self.p0_score = tk.Label(self.window, text="Score: 0", font=("Arial", 10), fg="green")
		self.p0_score.grid(row=0, column=0, columnspan=self.grid_size // 4, sticky="w")
		self.turn_label = tk.Label(self.window, text="Turns Left: 0", font=("Arial", 10))
		self.turn_label.grid(row=0, column=self.grid_size // 3, columnspan=self.grid_size // 3)
		self.p1_score = tk.Label(self.window, text="Score: 0", font=("Arial", 10), fg="blue")
		self.p1_score.grid(row=0, column=self.grid_size // 3 * 2, columnspan=self.grid_size // 4, sticky="w")
		print("[DEBUG] Creating grid buttons")
		for r in range(self.grid_size):
			for c in range(self.grid_size):
				btn = tk.Button(
					self.window,
					text=" ",
					width=3,
					height=2,
					font=("Arial", 12),
					command=lambda r=r, c=c: self.on_click(r, c)
				)
				btn.grid(row=r+1, column=c)
				self.buttons[r][c] = btn
		reset_btn = tk.Button(self.window, text="Reset", command=self.reset_game)
		reset_btn.grid(row=self.grid_size+1, column=0, columnspan=self.grid_size, sticky="we")
		print("[DEBUG] Reset button created")

	def on_click(self, row, col):
		print(f"[DEBUG] Button clicked at row={row}, col={col}")
		action = row * self.grid_size + col
		print(f"[DEBUG] Computed action={action}")
		self.step_game(action)
		self.player_selector = 1 - self.player_selector if self.player_selector is not None else 0
		print(f"[DEBUG] Next player selector={self.player_selector}")

		for x in range(self.grid_size):
			for y in range(self.grid_size):
				state = tk.NORMAL if self.legal_moves[self.player_selector][x][y] else tk.DISABLED
				self.buttons[x][y].config(state=state)
		print("[DEBUG] Button states updated based on legal moves")
		self.scores[0] = np.sum(self.grid & grid_values[0]["claim"])//grid_values[0]["claim"]
		self.scores[1] = np.sum(self.grid & grid_values[1]["claim"])//grid_values[1]["claim"]
		self.refresh_board_ui()
		if self.turn >= self.turn_limit:
			self.end_game()

	def end_game(self):
		for btn in self.buttons:
			for b in btn:
				b.config(state=tk.DISABLED)
		print("[DEBUG] Game over: turn limit reached")
		winner_string = "Player 0 wins!" if self.scores[0] > self.scores[1] else "Player 1 wins!" if self.scores[1] > self.scores[0] else "It's a draw!"
		self.turn_label.config(text=f"{winner_string}")

	def refresh_board_ui(self):
		default_bg = "SystemButtonFace"
		for x in range(self.grid_size):
			for y in range(self.grid_size):
				btn = self.buttons[x][y]
				cell = self.grid[x, y]
				if cell & grid_values[0]["claim"]:
					bg = "lightgreen"
				elif cell & grid_values[1]["claim"]:
					bg = "lightblue"
				else:
					bg = default_bg
				if cell & grid_values[0]["token"]:
					text, fg = "X", "green"
				elif cell & grid_values[1]["token"]:
					text, fg = "O", "blue"
				else:
					text, fg = "", "black"
				btn.config(bg=bg, text=text, fg=fg)
		self.p0_score.config(text=f"Score: {self.scores[0]}")
		self.p1_score.config(text=f"Score: {self.scores[1]}")
		self.turn_label.config(text=f"Turns Left: {self.turn_limit - self.turn}")
		print("[DEBUG] Board UI refreshed with claims and tokens")

	def reset_game(self):
		print("[DEBUG] Resetting game state")
		for row in self.buttons:
			for btn in row:
				if btn:
					btn.config(text=" ", state="normal")
		self.grid.fill(0)
		self.turn = 0
		self.player_selector = 0
		self.legal_moves = {
			0: np.ones((self.grid_size, self.grid_size), dtype=bool),
			1: np.ones((self.grid_size, self.grid_size), dtype=bool),
		}
		self.scores = [0, 0]
		self.refresh_board_ui()
		print("[DEBUG] Grid cleared, legal moves reset, turn=0, player_selector=0")

	def check_claim(self, player, row: int, col: int):
		print(f"[DEBUG] check_claim called for player={player}, row={row}, col={col}")
		token = grid_values[player]["token"]
		self.grid[row, col] |= token
		print(f"[DEBUG] Placed token. Grid[{row},{col}] now={self.grid[row,col]:04b}")

		mark_for_claim = np.zeros((self.grid_size, self.grid_size), dtype=bool)
		mark_for_remove = np.zeros((self.grid_size, self.grid_size), dtype=bool)

		# First horizontal
		step = (0, 1)
		start = (row, max(0, col - 2))
		end = (row, min(self.grid_size - 1, col + 2))
		cont = self._check_line(token, start, end, step)
		print(f"[DEBUG] Horizontal check result: {cont}")
		if cont["continuous"] >= 3:
			mark_for_remove[row, cont["start"][1]:cont["end"][1]+1] = True
			self.modify_claims(player, mark_for_claim, cont["start"], step)
			print("[DEBUG] Horizontal line claim/remove marked")

		# Second vertical
		step = (1, 0)
		start = (max(0, row - 2), col)
		end = (min(self.grid_size - 1, row + 2), col)
		cont = self._check_line(token, start, end, step)
		print(f"[DEBUG] Vertical check result: {cont}")
		if cont["continuous"] >= 3:
			mark_for_remove[cont["start"][0]:cont["end"][0]+1, col] = True
			self.modify_claims(player, mark_for_claim, cont["start"], step)
			print("[DEBUG] Vertical line claim/remove marked")

		# Third diagonal TL-BR
		step = (1, 1)
		candidates = [(row - 2, col - 2), (row - 1, col - 1), (row, col), (row + 1, col + 1), (row + 2, col + 2)]
		for i in range(len(candidates)):
			if self.out_of_bounds(candidates[4-i][0], candidates[4-i][1]):
				candidates.remove(candidates[4-i])
		start = candidates[0]
		end = candidates[-1]
		cont = self._check_line(token, start, end, step)
		print(f"[DEBUG] Diagonal TL-BR check result: {cont}")
		if cont["continuous"] >= 3:
			for i in range(cont["start"][0], cont["end"][0] + 1):
				mark_for_remove[i, i - cont["start"][0] + cont["start"][1]] = True
			self.modify_claims(player, mark_for_claim, cont["start"], step)
			print("[DEBUG] Diagonal TL-BR claim/remove marked")

		# Fourth diagonal TR-BL
		step = (1, -1)
		candidates = [(row - 2, col + 2), (row - 1, col + 1), (row, col), (row + 1, col - 1), (row + 2, col - 2)]
		for i in range(len(candidates)):
			if self.out_of_bounds(candidates[4-i][0], candidates[4-i][1]):
				candidates.remove(candidates[4-i])
		start = candidates[0]
		end = candidates[-1]
		cont = self._check_line(token, start, end, step)
		print(f"[DEBUG] Diagonal TR-BL check result: {cont}")
		if cont["continuous"] >= 3:
			for i in range(cont["start"][0], cont["end"][0] + 1):
				mark_for_remove[i, cont["start"][1] - (i - cont["start"][0])] = True
			self.modify_claims(player, mark_for_claim, cont["start"], step)
			print("[DEBUG] Diagonal TR-BL claim/remove marked")

		# Apply removals and claims with colored snapshot
		self.grid[mark_for_remove] = no_claim
		self.grid[mark_for_claim] &= (0b1111 - grid_values[1 - player]["claim"])
		self.grid[mark_for_claim] |= (grid_values[player]["claim"])
		print("[DEBUG] Applied removals and claims. Grid snapshot:")
		for r in range(self.grid_size):
			line = ""
			for c in range(self.grid_size):
				cell = self.grid[r, c]
				# background based on claim
				if cell & p0_claim:
					bg = BG_GREEN
				elif cell & p1_claim:
					bg = BG_BLUE
				else:
					bg = ''
				# token foreground
				value = format(self.grid[r, c], '04b')
				if cell & p0_token:
					fg = FG_GREEN + value
				elif cell & p1_token:
					fg = FG_BLUE + value
				else:
					fg = value
				line += f"{bg}{fg}{RESET} "
			print(line)

	def modify_claims(self, player, mark_for_claim, start, step):
		print(f"[DEBUG] modify_claims start={start}, step={step}, player={player}")
		curr = start
		op = 1
		while True:
			if self.out_of_bounds(curr[0], curr[1]) or (self.grid[curr[0], curr[1]] & grid_values[1-player]["token"]):
				if op > 0:
					op = -1
					curr = start
					print("[DEBUG] Reversing direction for modify_claims")
					continue
				break
			mark_for_claim[curr[0], curr[1]] = True
			curr = (curr[0] + op * step[0], curr[1] + op * step[1])
		print(f"[DEBUG] modify_claims marked positions:\n{mark_for_claim}")

	def out_of_bounds(self, row: int, col: int) -> bool:
		return row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size

	def _check_line(self, mask, start: Tuple[int, int], end: Tuple[int, int], step: Tuple[int, int]) -> dict:
		"""Internal contiguous-mask scanner with debug output."""
		print(f"[DEBUG] _check_line start={start}, end={end}, step={step}, mask={mask:04b}")
		# validation omitted for brevity
		max_continuous = continuous = 0
		max_start = max_end = curr_start = None
		last_mask = False
		curr = start
		_end = (end[0] + step[0], end[1] + step[1])
		while curr != _end:
			if self.out_of_bounds(curr[0], curr[1]):
				print(f"[DEBUG] Out of bounds at {curr}, breaking")
				break
			cell = self.grid[curr[0], curr[1]] & mask
			if cell:
				if last_mask:
					continuous += 1
				else:
					curr_start = curr
					continuous = 1
				last_mask = True
			else:
				if continuous > max_continuous:
					max_continuous = continuous
					max_start = curr_start
					max_end = (curr[0]-step[0], curr[1]-step[1])
				continuous = 0
				last_mask = False
			curr = (curr[0]+step[0], curr[1]+step[1])
		if continuous > max_continuous:
			max_continuous = continuous
			max_start = curr_start
			max_end = (curr[0]-step[0], curr[1]-step[1])
		result = {"start": max_start, "end": max_end, "continuous": max_continuous}
		print(f"[DEBUG] _check_line result={result}")
		return result

	def step_game(self, action):
		print(f"[DEBUG] step_game called with action={action}")
		player = self.player_selector
		self.turn += 1
		row, col = divmod(action, self.grid_size)
		print(f"[DEBUG] Decoded action to row={row}, col={col}, player={player}")
		if not self.legal_moves[player][row][col]:
			print(f"[ERROR] Invalid move by player {player} at ({row}, {col})")
			return
		self.check_claim(player, row, col)
		occupied = (self.grid & (p0_token | p1_token)) != 0
		c0 = (self.grid & p0_claim) != 0
		c1 = (self.grid & p1_claim) != 0
		self.legal_moves[0] = ~occupied & ~c1
		self.legal_moves[1] = ~occupied & ~c0
		print(f"[DEBUG] Updated legal_moves for both players")

	def run(self):
		print("[DEBUG] Running mainloop")
		self.window.mainloop()

if __name__ == "__main__":
	game = PopuGameGUI(size=_DEFAULT_SIZE, turn_limit=_DEFAULT_TURN_LIMIT)
	game.run()
