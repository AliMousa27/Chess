from .linear_pieces import Linear_Piece
from .piece_color import Piece_Color
from typing import List, Tuple

class Bishop(Linear_Piece):
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int) -> None:
    super().__init__(name,position,color,img_size,[(1, 1), (1, -1), (-1, 1), (-1, -1)]  )
    
  



      
