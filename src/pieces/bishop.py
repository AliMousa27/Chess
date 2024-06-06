from piece import Piece
from piece_color import Piece_Color
from typing import List, Tuple

class Bishop(Piece):
  def __init__(self,name:str,init_position:tuple,color: Piece_Color) -> None:
    super().__init__(name,init_position,color)
  
  def calc_all_moves(self) -> List[Tuple]:
    print("d")
      
  def move(self) -> None:
    print("d")