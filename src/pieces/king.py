from .piece import Piece
from .piece_color import Piece_Color
from typing import List, Tuple

class King(Piece):
  """
  Represents a King chess piece.

  Attributes:
    name (str): The name of the piece.
    position (tuple): The current position of the piece on the chessboard.
    color (Piece_Color): The color of the piece.
    img_size (int): The size of the piece's image.

  Methods:
    calc_all_moves(): Calculates all possible moves for the King.
    filter_moves(board, is_pinned, check_for_pins): Filters the possible moves based on the current board state.
    can_swap_with_rook(row, rook_has_stepped): Checks if the King can swap places with a Rook for castling.
    can_move_to_square(board, row, col, check_for_pins, is_pinned): Checks if the King can move to a specific square on the board.

  """

  def __init__(self, name: str, position: tuple, color: Piece_Color, img_size: int) -> None:
    super().__init__(name, position, color, img_size)
    self.has_stepped = False

  def calc_all_moves(self) -> List[Tuple[int, int]]:
    """
    Calculates all possible moves for the King.

    Returns:
      List[Tuple[int, int]]: A list of tuples representing the possible moves.

    """
    row, col = self.position
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (0, -1), (-1, 0), (1, 0), (0, 1)]
    # return all the possible moves within the bounds
    return [(dr + row, dc + col) for dr, dc in directions if 0 <= dr + row <= 7 and 0 <= dc + col <= 7]

  def filter_moves(self, board: List[List], is_pinned, check_for_pins: bool):
    """
    Filters the possible moves based on the current board state.

    Args:
      board (List[List]): The current state of the chessboard.
      is_pinned: A function that checks if a piece is pinned.
      check_for_pins (bool): Flag indicating whether to check for pins.

    Returns:
      List[Tuple[int, int]]: A list of tuples representing the filtered moves.

    """
    moves = [(row, col) for row, col in self.calc_all_moves() if self.can_move_to_square(board, row, col, check_for_pins, is_pinned)]
    # castling move TODO
    if not self.has_stepped:
      pass
    return moves

  # TODO

  def can_swap_with_rook(self, row, rook_has_stepped: bool) -> bool:
    """
    Checks if the King can swap places with a Rook for castling.

    Args:
      row: The row of the Rook.
      rook_has_stepped (bool): Flag indicating whether the Rook has moved.

    Returns:
      bool: True if the King can swap places with the Rook, False otherwise.

    """
    # we dont check if the king has moved because game class does that
    can_castle = all(row[i].occupant is None for i in range(5, 7)) and not rook_has_stepped
    return can_castle

  def can_move_to_square(self, board: List[List], row: int, col: int, check_for_pins: bool, is_pinned) -> bool:
    """
    Checks if the King can move to a specific square on the board.

    Args:
      board (List[List]): The current state of the chessboard.
      row (int): The row of the square.
      col (int): The column of the square.
      check_for_pins (bool): Flag indicating whether to check for pins.
      is_pinned: A function that checks if a piece is pinned.

    Returns:
      bool: True if the King can move to the square, False otherwise.

    """
    return (not board[row][col].occupant or board[row][col].occupant.color != self.color) and (not check_for_pins or not is_pinned(self, (row, col)))