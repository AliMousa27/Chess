from .piece import Piece
from .piece_color import Piece_Color
from typing import List, Tuple
from typing import Callable
class Knight(Piece):
  def __init__(self, name: str, position: tuple, color: Piece_Color, img_size: int) -> None:
    super().__init__(name, position, color, img_size)
  
  def calc_all_moves(self) -> List[Tuple]:
    """
    Returns all the possible moves for the knight.

    Args:
        None

    Returns:
        A list of all possible moves for the knight.
    """
    row, col = self.position
    directions = [(-1, 2), (1, 2), (-2, 1), (2, 1), (-2, 1), (-2, -1), (2, -1), (-1, -2), (1, -2)]
    return [(dr + row, dc + col) for dr, dc in directions if 0 <= dr + row <= 7 and 0 <= dc + col <= 7]

      
  def filter_moves(self, board: List[List], is_pinned, check_for_pins: bool) -> List[Tuple]:
    """
    Filters the possible moves for the knight based on the current board state.

    Args:
        board: The current chess board represented as a 2D list.
        is_pinned: A function that checks if a piece is pinned.
        check_for_pins: A flag indicating whether to check for pins.

    Returns:
        A list of filtered moves for the knight.
    """
    return [(row, col) for row, col in self.calc_all_moves() if self.can_move_to_square(row, col, board, check_for_pins, is_pinned)]

  def can_move_to_square(self, row, col, board, check_for_pins, is_pinned) -> bool:
    """
    Checks if the knight can move to the specified square on the board.

    Args:
        row: The row index of the square.
        col: The column index of the square.
        board: The current chess board represented as a 2D list.
        check_for_pins: A flag indicating whether to check for pins.
        is_pinned: A function that checks if a piece is pinned.

    Returns:
        True if the knight can move to the square, False otherwise.
    """
    return (not board[row][col].occupant or board[row][col].occupant.color != self.color) and (not check_for_pins or not is_pinned(self, (row, col)))