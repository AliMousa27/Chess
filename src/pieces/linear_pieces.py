from abc import ABCMeta, abstractmethod
from pieces.piece import Piece
from typing import List, Tuple
from .piece_color import Piece_Color

class Linear_Piece(Piece, metaclass=ABCMeta):
  @abstractmethod
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int, directions:List[Tuple[int,int]]) -> None:
    super().__init__(name,position,color,img_size)
    self.directions = directions
    
  def calc_all_moves(self) -> List[Tuple]:
    return self.directions

  def filter_moves(self, board: List[List], is_pinned, check_for_pins: bool):
      # row col direct
      row, col = self.position
      skip_direction = False
      moves = []
      for dr, dc in self.directions:
          for i in range(1, 9):
              new_row = row + i * dr
              new_col = col + i * dc
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