"""
Complete the methods below to implement your AI player.
You may add helper methods and classes as needed.

DO NOT modify the method signatures of __init__ or get_move.
"""

import random, time
from typing import Tuple, List, Dict

class Player:
    """
    Your AI player for Breakthrough.
    """
    
    def __init__(self, player_number: int): # DO NOT MODIFY THIS LINE
        """
        Initialize your AI.
        
        Args:
            player_number: 1 (White, starts at bottom) or 2 (Black, starts at top)
        """
        self.player_number = player_number
        self.opponent_number = 3 - player_number
        
        # Transposition table for caching evaluations
        self.transposition_table = {}
        self.start_time = 0
        self.time_limit = 0
    
    def get_move(self, state: Dict, time_limit: float) -> Tuple[int, int, int, int]: # DO NOT MODIFY THIS LINE
        """
        Get the next move for your AI.
        
        This method will be called by the game engine. You must return a valid move
        within the time limit.
        
        Args:
            state: Dictionary containing:
                - 'board': 8x8 list of lists, where:
                    0 = empty
                    1 = Player 1 (White)
                    2 = Player 2 (Black)
                - 'current_player': 1 or 2
                - 'move_count': Number of moves played so far
            time_limit: Maximum time allowed in seconds
        
        Returns:
            Tuple (from_row, from_col, to_row, to_col) representing your move
        
        Example:
            return (1, 3, 2, 3)  # Move piece from (1,3) to (2,3)
        """
        legal_moves = self.get_legal_moves(state['board'], self.player_number) # gets all legal moves
        
        if not legal_moves:
            raise ValueError("No legal moves available!") # returns an error if there are no legal moves
        
        # Check for immediate winning move
        for move in legal_moves:
            new_board = self.apply_move(state['board'], move, self.player_number)
            if self.has_won(new_board, self.player_number):
                return move
        
        # Initialize timing for iterative deepening
        self.start_time = time.time()
        self.time_limit = time_limit * 0.75  # Use 75% of time limit for safety
        
        # Iterative deepening - try increasing depths until time runs out
        best_move = random.choice(legal_moves)
        max_depth = 10  # Maximum depth to search
        
        for depth in range(1, max_depth + 1):
            time_used = time.time() - self.start_time
            # Estimate if we have enough time for next iteration
            # Each depth roughly takes 3-4x longer than previous
            # Stop early if we've used 25% of time budget
            if time_used > self.time_limit * 0.25:
                break
            
            try:
                #calls maximizing move function to get best move at current depth
                score, move = self.make_max_move(state['board'], depth=depth, alpha=float('-inf'), beta=float('inf'))
                if move is not None:
                    best_move = move # updates best move found so far
                # If we found a winning move, return it immediately
                if score >= 100000:
                    break
            except TimeoutError:
                break  # Time limit exceeded
        
        return best_move # returns the best move found within time limit
    
    # =========================================================================
    # HELPER METHODS - Implement these to build your AI
    # =========================================================================
    
    def get_legal_moves(self, board: List[List[int]], player: int) -> List[Tuple[int, int, int, int]]:
        """
        Get all legal moves for a player.
        
        Args:
            board: 8x8 board state
            player: Player number (1 or 2)
        
        Returns:
            List of moves as tuples (from_row, from_col, to_row, to_col)
        """
        moves = [] # Initialize empty move list
        direction = 1 if player == 1 else -1 # Direction of movement based on player
        
        for row in range(8): # Iterate through each row
            for col in range(8): # Iterate through each column
                if board[row][col] == player: # If the cell has a pawn of the current player
                    new_row = row + direction # Move one row forward
                    
                    if 0 <= new_row < 8: # check if new row is withing bounds
                        # Straight forward (only if empty)
                        if board[new_row][col] == 0:
                            moves.append((row, col, new_row, col))
                        
                        # Diagonal left
                        if col > 0 and board[new_row][col - 1] != player:
                            moves.append((row, col, new_row, col - 1))
                        
                        # Diagonal right
                        if col < 7 and board[new_row][col + 1] != player:
                            moves.append((row, col, new_row, col + 1))
        
        return moves # return list of legal moves
    
    def order_moves(self, board: List[List[int]], moves: List[Tuple[int, int, int, int]], player: int) -> List[Tuple[int, int, int, int]]:
        """
        Order moves for better alpha-beta pruning.
        Prioritize: captures, forward advancement, center control

        Args:
            board: 8x8 board state
            moves: List of legal moves
            player: Player number (1 or 2)

        Returns:
            List of moves ordered by priority
        """
        def move_score(move): 
            """
            Score a move based on heuristics

            Args:
                move: Tuple (from_row, from_col, to_row, to_col)

            Returns:
                Score as an integer
            """
            from_row, from_col, to_row, to_col = move # unpack move
            score = 0 # initialize score
            
            # Prioritize captures
            if board[to_row][to_col] != 0:
                score += 10000
            
            # Prioritize forward advancement
            if player == 1:
                score += to_row * 100
            else:
                score += (7 - to_row) * 100
            
            # Prefer center columns
            center_distance = abs(to_col - 3.5)
            score += (3.5 - center_distance) * 10
            
            # Bonus for pawns very close to promotion
            if player == 1 and to_row >= 6:
                score += 5000
            elif player == 2 and to_row <= 1:
                score += 5000
            
            return score # return computed score
        
        return sorted(moves, key=move_score, reverse=True) # sort moves by score descending
    def has_won(self, board: List[List[int]], player: int) -> bool:
        """
        Check if the given player has won the game.
        
        Args:
            board: 8x8 board state
            player: Player number (1 or 2)
        
        Returns:
            True if the player has won, False otherwise
        """
        target_row = 7 if player == 1 else 0 # Row to check for victory
        for col in range(8):
            if board[target_row][col] == player: # checks if any pawn has reached the opponent's back row by column
                return True
        return False


    
    def make_max_move(self, board: List[List[int]], depth, alpha: int, beta: int) -> List[List[int]]:
        """
        Apply a maximizing move to the board and return the new board state.
        Uses minimax algorithm with alpha-beta pruning to select the best move.
        
        Args:
            board: 8x8 board state
            alpha: Alpha value for pruning
            Beta: Beta value for pruning

        Returns:
            New board state after the move
        """
        # Check time limit
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded")
        
        # Check transposition table
        board_hash = self.hash_board(board) # hash the board state
        if board_hash in self.transposition_table: # if board state is cached
            cached_depth, cached_value, cached_move = self.transposition_table[board_hash] # retrieve the cached depth, value, and move
            if cached_depth >= depth: # if cached depth is sufficient
                return cached_value, cached_move # return cached value and move

        legal_moves = self.get_legal_moves(board, self.player_number) # gets legal moves for maximizing player
        
        # Terminal state check
        if depth == 0 or self.has_won(board, self.player_number) or self.has_won(board, self.opponent_number) or not legal_moves:
            return self.utility(board), None

        best_value = float('-inf') # initialize best value
        best_move = None # initialize best move

        # Order moves for better pruning
        ordered_moves = self.order_moves(board, legal_moves, self.player_number)

        for move in ordered_moves:
            new_board = self.apply_move(board, move, self.player_number) # apply the move
            
            # Check for winning move
            if self.has_won(new_board, self.player_number):
                return 1000000, move # Immediate win

            value, _ = self.make_min_move(new_board, depth - 1, alpha, beta) # call minimizing move
            if value > best_value:  # update best value and move if current value is better
                best_value = value
                best_move = move
            alpha = max(alpha, best_value) # update alpha
            if alpha >= beta:
                break  # Beta cutoff

        # Store in transposition table
        self.transposition_table[board_hash] = (depth, best_value, best_move)
        
        return best_value, best_move # return best value and move

    def make_min_move(self, board: List[List[int]], depth, alpha: int, beta: int) -> List[List[int]]:
        """
        Apply a minimizing move to the board and return the new board state.
        Uses minimax algorithm with alpha-beta pruning to select the best move.
        
        Args:
            board: 8x8 board state
            alpha: Alpha value for pruning
            Beta: Beta value for pruning

        Returns:
            New board state after the move
        """
        legal_moves = self.get_legal_moves(board, self.opponent_number) # gets legal moves for minimizing player
        
        # Check time limit
        if depth == 0 or self.has_won(board, self.player_number) or self.has_won(board, self.opponent_number) or not legal_moves:
            return self.utility(board), None
        
        best_value = float('inf') # initialize best value
        best_move = None # initialize best move
        
        # Order moves for better pruning
        ordered_moves = self.order_moves(board, legal_moves, self.opponent_number)
        
        for move in ordered_moves:
            new_board = self.apply_move(board, move, self.opponent_number) # apply the move
            
            # If opponent move instantly wins (bad for us), return loss!
            if self.has_won(new_board, self.opponent_number):
                return -1000000, move # Immediate loss
            
            value, _ = self.make_max_move(new_board, depth - 1, alpha, beta) # call maximizing move
            if value < best_value: # update best value and move if current value is better
                best_value = value 
                best_move = move
            beta = min(beta, best_value) # update beta
            if alpha >= beta: 
                break  # Alpha cutoff
        
        return best_value, best_move # return best value and move

    def apply_move(self, board: List[List[int]], move: Tuple[int, int, int, int], player: int) -> List[List[int]]:
        """
        Apply a move to the board and return the new board state.
        
        Args:
            board: 8x8 board state
            move: Tuple (from_row, from_col, to_row, to_col)
            player: Player number (1 or 2)
        Returns:
            New board state after the move
        """
        from_row, from_col, to_row, to_col = move # unpack move
        new_board = [row[:] for row in board]  # Deep copy of the board
        new_board[to_row][to_col] = player # place player's pawn at new position
        new_board[from_row][from_col] = 0 # empty the original position
        return new_board # return new board state
    
    def hash_board(self, board: List[List[int]]) -> int:
        """
        Create a hash of the board state for transposition table.

        Args:
            board: 8x8 board state
        
        Returns:
            Hash value of the board state
        """
        return hash(tuple(tuple(row) for row in board))

    def utility(self, board):
        """
        Evaluate the board position from the perspective of the AI player.
        Higher values are better for the AI.

        Args:
            board: 8x8 board state

        Returns:
            Utility score as an integer
        """
        if self.has_won(board, self.player_number):
            return 1000000 # AI win
        if self.has_won(board, self.opponent_number):
            return -1000000 # Opponent win

        my_pawns, opp_pawns = 0, 0 # Count of pawns
        my_adv, opp_adv = 0, 0 # Total advancement
        my_adv_sq, opp_adv_sq = 0, 0  # Squared advancement for exponential reward
        my_mobility, opp_mobility = 0, 0 # Mobility counts
        my_center_control, opp_center_control = 0, 0 # Center control counts
        
        # Count pawns and advancement
        for row in range(8):
            for col in range(8):
                cell = board[row][col]
                if cell == self.player_number:
                    my_pawns += 1
                    # Calculate advancement (distance toward goal)
                    if self.player_number == 1:
                        adv = row  # Player 1 wants to reach row 7
                    else:
                        adv = (7 - row)  # Player 2 wants to reach row 0
                    my_adv += adv
                    my_adv_sq += adv * adv  # Reward getting closer to goal exponentially
                    
                    # Center control bonus (columns 3 and 4 are center)
                    if 2 <= col <= 5:
                        my_center_control += 1
                        
                elif cell == self.opponent_number:
                    opp_pawns += 1
                    # Calculate opponent advancement
                    if self.opponent_number == 1:
                        adv = row
                    else:
                        adv = (7 - row)  # Player 2 wants to reach row 0
                    opp_adv += adv # opponent advancement
                    opp_adv_sq += adv * adv # opponent squared advancement
                    
                    # Center control for opponent
                    if 2 <= col <= 5:
                        opp_center_control += 1
        
        # Mobility (number of legal moves)
        my_mobility = len(self.get_legal_moves(board, self.player_number))
        opp_mobility = len(self.get_legal_moves(board, self.opponent_number))

        # Strong penalty if opponent is close to winning
        threat_penalty = 0
        if self.player_number == 1:
            # Check if opponent (player 2) is close to row 0
            for col in range(8):
                if board[1][col] == self.opponent_number:
                    threat_penalty -= 80000
                elif board[2][col] == self.opponent_number:
                    threat_penalty -= 15000
        else:
            # Check if opponent (player 1) is close to row 7
            for col in range(8):
                if board[6][col] == self.opponent_number:
                    threat_penalty -= 80000
                elif board[5][col] == self.opponent_number:
                    threat_penalty -= 15000

        # Bonus for my pawns being close to goal
        my_proximity_bonus = 0
        if self.player_number == 1: # Player 1
            for col in range(8): # for each column
                if board[6][col] == self.player_number: # if pawn is in row 6
                    my_proximity_bonus += 30000 # big bonus for being one step away
                elif board[5][col] == self.player_number: # if pawn is in row 5
                    my_proximity_bonus += 8000 # smaller bonus for being two steps away
        else:
            for col in range(8): # Player 2
                if board[1][col] == self.player_number:
                    my_proximity_bonus += 30000
                elif board[2][col] == self.player_number:
                    my_proximity_bonus += 8000

        # Pawn structure: penalize isolated pawns (no friendly pawns adjacent)
        my_structure, opp_structure = 0, 0
        for row in range(8):
            for col in range(8):
                if board[row][col] == self.player_number: # check my pawns
                    has_support = False # flag for adjacent friendly pawns
                    for dc in [-1, 1]: # check left and right
                        if 0 <= col + dc < 8 and board[row][col + dc] == self.player_number: # adjacent friendly pawn found
                            has_support = True # adjacent friendly pawn found
                            break # no need to check further
                    if has_support: # adjacent friendly pawn found
                        my_structure += 20 # reward for supported pawn
                elif board[row][col] == self.opponent_number: # check opponent pawns
                    has_support = False # adjacent friendly pawns
                    for dc in [-1, 1]: # check left and right
                        if 0 <= col + dc < 8 and board[row][col + dc] == self.opponent_number: # adjacent friendly pawn found
                            has_support = True # adjacent friendly pawn found
                            break # no need to check further
                    if has_support: # adjacent friendly pawn found
                        opp_structure += 20 # reward for supported pawn

        return (
            500 * (my_pawns - opp_pawns) + # pawn count difference (weighted high because losing pawns is bad)
            100 * (my_adv - opp_adv) + # advancement difference (weighted high because advancing pawns is good)
            50 * (my_adv_sq - opp_adv_sq) + # squared advancement difference (weighted to emphasize advanced pawns)
            10 * (my_mobility - opp_mobility) + # mobility difference (weighted to encourage flexibility)
            30 * (my_center_control - opp_center_control) + # center control difference (weighted to encourage center control)
            (my_structure - opp_structure) + # pawn structure difference (weighted to encourage pawn support)
            my_proximity_bonus + # bonus for proximity to goal
            threat_penalty # penalty for opponent threats
        )
