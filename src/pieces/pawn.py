from .piece import Piece
from .piece_color import Piece_Color
from typing import List, Tuple

class Pawn(Piece):
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int) -> None:
    super().__init__(name,position,color,img_size)
  
  def calc_all_moves(self) -> List[Tuple]:
    print("d")
      
  def move(self) -> None:
    print("d")