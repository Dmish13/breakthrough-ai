"""
Visual Game Player for Breakthrough
Watch AI vs AI games with nice formatting and move-by-move display

Usage:
    python play_game_visual.py                                  # Solution vs Greedy
    python play_game_visual.py --p1 solution --p2 advanced      # Custom matchup
    python play_game_visual.py --fast                           # No delays between moves
"""

import argparse
import time
import sys
import importlib.util

def load_module_from_file(file_path):
	"""Dynamically load a Python module from file path."""
	spec = importlib.util.spec_from_file_location(file_path, file_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

class BreakthroughBoard:
	"""
	Board representation for Breakthrough game.

	Board layout (8x8):
	- Row 0 is Player 2's goal (White's goal)
	- Row 7 is Player 1's goal (Black's goal)
	- Player 1 (White) starts at rows 0-1
	- Player 2 (Black) starts at rows 6-7
	"""
	EMPTY = 0
	PLAYER1 = 1  # White
	PLAYER2 = 2  # Black

	def __init__(self):
		"""Initialize board with starting position."""
		self.current_player = self.PLAYER1
		self.move_count = 0
		
		self.board = [[self.EMPTY for _ in range(8)] for _ in range(8)]
		for col in range(8):
			self.board[0][col] = self.PLAYER1
			self.board[1][col] = self.PLAYER1
			self.board[6][col] = self.PLAYER2
			self.board[7][col] = self.PLAYER2

	def get_legal_moves(self, player=None):
		"""
		Get all legal moves for the specified player.
		
		Args:
			player: Player number (1 or 2). If None, uses current_player.
		
		Returns:
			List of tuples: [(from_row, from_col, to_row, to_col), ...]
		"""
		if player is None:
			player = self.current_player
		
		moves = []
		direction = 1 if player == self.PLAYER1 else -1
		
		for row in range(8):
			for col in range(8):
				if self.board[row][col] == player:
					# Forward move
					new_row = row + direction
					if 0 <= new_row < 8:
						# Straight forward (only if empty)
						if self.board[new_row][col] == self.EMPTY:
							moves.append((row, col, new_row, col))
						
						# Diagonal forward left
						if col > 0:
							target = self.board[new_row][col - 1]
							if target != player:  # Can move to empty or capture
								moves.append((row, col, new_row, col - 1))
						
						# Diagonal forward right
						if col < 7:
							target = self.board[new_row][col + 1]
							if target != player:  # Can move to empty or capture
								moves.append((row, col, new_row, col + 1))
		
		return moves

	def make_move(self, move):
		"""
		Execute a move on the board.
		
		Args:
			move: Tuple (from_row, from_col, to_row, to_col)
		
		Returns:
			True if move was successful, False otherwise
		"""
		from_row, from_col, to_row, to_col = move
		
		# Validation
		if not (0 <= from_row < 8 and 0 <= from_col < 8 and 
				0 <= to_row < 8 and 0 <= to_col < 8):
			return False
		
		if self.board[from_row][from_col] != self.current_player:
			return False
		
		# Check if move is legal
		if move not in self.get_legal_moves():
			return False
		
		# Execute move
		self.board[to_row][to_col] = self.current_player
		self.board[from_row][from_col] = self.EMPTY
		
		# Switch players
		self.current_player = 3 - self.current_player  # Switches 1<->2
		self.move_count += 1
		
		return True

	def is_terminal(self):
		"""
		Check if the game is over.
		
		Returns:
			True if game is over, False otherwise
		"""
		# Check if Player 1 reached row 7
		for col in range(8):
			if self.board[7][col] == self.PLAYER1:
				return True
		
		# Check if Player 2 reached row 0
		for col in range(8):
			if self.board[0][col] == self.PLAYER2:
				return True
		
		# Check if current player has no legal moves
		if len(self.get_legal_moves()) == 0:
			return True
		
		return False

	def get_winner(self):
		"""
		Determine the winner of the game.
		
		Returns:
			1 if Player 1 wins
			2 if Player 2 wins
			0 if game is not over or is a draw
		"""
		# Check breakthrough wins
		for col in range(8):
			if self.board[7][col] == self.PLAYER1:
				return self.PLAYER1
			if self.board[0][col] == self.PLAYER2:
				return self.PLAYER2
		
		# Check if current player has no moves (loses)
		if len(self.get_legal_moves()) == 0:
			return 3 - self.current_player  # Opponent wins
		
		return 0  # Game not over

	def get_state_dict(self):
		"""
		Get board state as dictionary (for student AI interface).
		
		Returns:
			Dictionary with board state information
		"""
		return {
			'board': [row[:] for row in self.board],
			'current_player': self.current_player,
			'move_count': self.move_count
		}

	def display(self):
		"""Print the board with nice formatting"""
		print("\n  | " + " ".join([str(i) for i in range(8)]))
		print("----" + "-" * 15)
		
		for i in range(7, -1, -1):
			row_str = f"{i} | "
			for j in range(8):
				piece = self.board[i][j]
				if piece == 1:
					row_str += "W "  # White (Player 1)
				elif piece == 2:
					row_str += "B "  # Black (Player 2)
				else:
					row_str += "· "  # Empty
			print(row_str)
		
		print()

def format_move(move):
    """Format move in readable way."""
    from_row, from_col, to_row, to_col = move
    return f"({from_row},{from_col}) → ({to_row},{to_col})"

def play_game(player1_ai, player2_ai, player1_name="Player 1", player2_name="Player 2", time_limit=5.0):
	"""
	Play a game with visual output.

	Args:
		player1_ai: AI instance for Player 1 (White)
		player2_ai: AI instance for Player 2 (Black)
		player1_name: Name for display
		player2_name: Name for display
		time_limit: Time limit per move
	"""
	board = BreakthroughBoard()
	move_count = 0

	print("\n" + "="*60)
	print("BREAKTHROUGH GAME")
	print("="*60)
	print(f"Player 1 (White): {player1_name}")
	print(f"Player 2 (Black): {player2_name}")
	print("="*60)

	print("\nStarting position:")
	board.display()

	winner = None
	while not board.is_terminal():
		current_player = board.current_player
		current_ai = player1_ai if current_player == 1 else player2_ai
		current_name = player1_name if current_player == 1 else player2_name
		
		print(f"\n{'='*60}")
		print(f"Move {move_count + 1}: {current_name}'s turn")
		print(f"{'='*60}")
		
		# Get move from AI
		try:
			start_time = time.time()
			state = board.get_state_dict()
			move = current_ai.get_move(state, time_limit)
			elapsed = time.time() - start_time
			if elapsed > time_limit * 1.2:
				print(f"TIME LIMIT EXCEEDED! ({elapsed:.2f}s)")
				winner = 3 - current_player
				break
		except Exception as e:
			print(f"ERROR: {e}")
			winner = 3 - current_player
			break
		
		# Display move info
		print(f"Move: {format_move(move)}")
		print(f"Time: {elapsed:.3f}s")
		
		# Make move
		if not board.make_move(move):
			print(f"ILLEGAL MOVE!")
			winner = 3 - current_player
			break
		
		# Display board
		board.display()
		
		# Check for breakthrough
		if any(board.board[7][col] == 1 for col in range(8)):
			winner = 1
			break
		elif any(board.board[0][col] == 2 for col in range(8)):
			winner = 2
			break
		
		move_count += 1

	# Game over
	print("\n" + "="*60)
	print("GAME OVER")
	print("="*60)

	if winner is None:
		winner = board.get_winner()

	if winner == 1:
		print(f"Winner: {player1_name}")
	elif winner == 2:
		print(f"Winner: {player2_name}")
	else:
		print("Draw")

	print(f"Total moves: {move_count}")
	print("="*60 + "\n")

def main():
	parser = argparse.ArgumentParser(
		description='Watch a Breakthrough game',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python Play.py --p1 BreakthroughAI.py --p2 Opponent.py
  python Play.py --p1 Opponent.py --p2 BreakthroughAI.py
"""
	)

	parser.add_argument('--p1', type=str, default=None,
						help='Player 1 file')
	parser.add_argument('--p2', type=str, default=None,
						help='Player 2 file')
	args = parser.parse_args()

	try:
		p1_module = load_module_from_file(args.p1)
		p1 = p1_module.Player
		player1_ai = p1(1)
		player1_name = args.p1
	except Exception as e:
		print(f"Error loading Player 1: {e}")
		return

	try:
		p2_module = load_module_from_file(args.p2)
		p2 = p2_module.Player
		player2_ai = p2(2)
		player2_name = args.p2
	except Exception as e:
		print(f"Error loading Player 2: {e}")
		return

	play_game(player1_ai, player2_ai, player1_name, player2_name)

if __name__ == "__main__":
    main()