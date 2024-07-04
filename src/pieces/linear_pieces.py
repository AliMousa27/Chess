from abc import ABCMeta, abstractmethod
from pieces.piece import Piece
from typing import List, Tuple
from .piece_color import Piece_Color

class Linear_Piece(Piece, metaclass=ABCMeta):
  @abstractmethod
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int, directions:List[Tuple[int,int]]) -> None:
    """
    Constructor for the Linear_Piece class
    Args:
      name: str - name of the piece
      position: tuple - the position of the piece on the board
      color: Piece_Color - the color of the piece
      img_size: int - the size of the image to be displayed
      directions: List[Tuple[int,int]] - the directions the piece can move in
    Return:
        None  
    """
    super().__init__(name,position,color,img_size)
    self.directions = directions
    
  def calc_all_moves(self) -> List[Tuple]:
    return self.directions


  def filter_moves(self, board: List[List], is_pinned, check_for_pins: bool)-> List[Tuple[int,int]]:
    """
    Method to filter the possible moves for the piece based on the current board state.
    Args:
			board: List[List] - the current chess board represented as a 2D list
			is_pinned: Callable - a function that checks if a piece is pinned
			check_for_pins: bool - a flag indicating whether to check for pins
		Returns:
			List[Tuple] - a list of filtered moves for the piece
    """
    # row col direct
    row, col = self.position
    skip_direction = False
    moves:List[Tuple[int,int]] = []
    # for each direction in row and col
    for dr, dc in self.directions:
        # for each square in the direction
        for i in range(1, 9):
            new_row = row + i * dr
            new_col = col + i * dc
            # if the square is out of bounds then skip that direction
            if not 0 <= new_row <= 7 or not 0 <= new_col <= 7:
              continue
            destination_square = board[new_row][new_col]
            # if its occupied with the same color then skip that direction 
            if destination_square.occupant and destination_square.occupant.color == self.color:
              skip_direction = True
              break
            if check_for_pins and is_pinned(self, (new_row, new_col)):
              continue
            elif destination_square.occupant and destination_square.occupant.color != self.color:
              moves.append((new_row, new_col))
              skip_direction = True
              break
            elif not destination_square.occupant:
                moves.append((new_row, new_col))
            if skip_direction:
              skip_direction = False
              continue
    return moves