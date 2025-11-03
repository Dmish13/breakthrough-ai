import random
from typing import Tuple, List, Dict

class Player:
	def __init__(self, player_number: int):
		self.player_number = player_number

	def get_legal_moves(self, board: List[List[int]], player: int) -> List[Tuple[int, int, int, int]]:
		moves = []
		direction = 1 if player == 1 else -1
		for row in range(8):
			for col in range(8):
				if board[row][col] == player:
					new_row = row + direction
					if 0 <= new_row < 8:
						if board[new_row][col] == 0: moves.append((row, col, new_row, col))
						if col > 0 and board[new_row][col - 1] != player: moves.append((row, col, new_row, col - 1))
						if col < 7 and board[new_row][col + 1] != player: moves.append((row, col, new_row, col + 1))
		return moves

	def get_move(self, state: Dict, time_limit: float) -> Tuple[int, int, int, int]:
		moves = self.get_legal_moves(state['board'], self.player_number)
		return random.choice(moves)
