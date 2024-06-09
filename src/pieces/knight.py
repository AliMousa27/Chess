from .piece import Piece
from .piece_color import Piece_Color
from typing import List, Tuple

class Knight(Piece):
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int) -> None:
    super().__init__(name,position,color,img_size)
  
  def calc_all_moves(self) -> List[Tuple]:
    row,col = self.position
    directions = [(-1,2),(1,2),(-2,1),(2,1),(-2,1),(-2,-1),(2,-1),(-1,-2),(1,-2)]
    return [(dr+row,dc+col) for dr,dc in directions if 0 <= dr+row <= 7 and 0 <= dc+col <= 7]

      
  def move(self) -> None:
    print("d")