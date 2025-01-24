"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count the number of X's and O's on the board
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    # Determine which player's turn it is
    if x_count > o_count:
        return O  # O's turn if X has played more
    else:
        return X  # X's turn if they have played equally or less


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if terminal(board):
        return set()  # No actions possible on a terminal board

    return {(i, j) for i in range(3) for j in range(3) if board[i][j] is EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action

    # Check if the indices are within bounds
    if not (0 <= i < 3 and 0 <= j < 3):
        raise ValueError("Invalid action: Indices are out of bounds.")

    # Check if the cell is already occupied
    if board[i][j] is not EMPTY:
        raise ValueError("Invalid action: Cell is already occupied.")
    
    # Create a deep copy of the board to avoid modifying the original
    new_board = [row[:] for row in board]
    new_board[i][j] = player(board)
    
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows for a winner
    for row in board:
        if row[0] == row[1] == row[2] != EMPTY:
            return row[0]

    # Check columns for a winner
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != EMPTY:
            return board[0][col]

    # Check diagonals for a winner
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None  # No winner


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if there is a winner
    if winner(board) is not None:
        return True
    
    # Check if the board is full (no empty cells)
    if all(cell is not EMPTY for row in board for cell in row):
        return True
    
    return False  # Game is still in progress


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    
    if win == X:
        return 1  # X wins
    elif win == O:
        return -1  # O wins
    else:
        return 0  # Tie


def max_value(board):
    """
    Compute the maximum utility value that can be achieved from the given board state.
    """
    # Check if the game is over
    if terminal(board):
        return utility(board), None

    v = float('-inf')
    best_action = None
    for action in actions(board):
        min_val = min_value(result(board, action))[0]  # Get only the score
        if min_val > v:
            v = min_val
            best_action = action
    return v, best_action


def min_value(board):
    """
    Compute the minimum utility value that can be achieved from the given board state.
    """
    # Check if the game is over
    if terminal(board):
        return utility(board), None

    v = float('inf')
    best_action = None
    for action in actions(board):
        max_val = max_value(result(board, action))[0]  # Get only the score
        if max_val < v:
            v = max_val
            best_action = action
    return v, best_action

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None  # No moves available

    current_player = player(board)

    if current_player == X:
        _, action = max_value(board)  # Maximize for X
    else:
        _, action = min_value(board)  # Minimize for O

    return action  # Return the optimal action (i, j)
