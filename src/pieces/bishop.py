from .piece import Piece
from .piece_color import Piece_Color
from typing import List, Tuple
from pprint import pprint
class Bishop(Piece):
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int) -> None:
    super().__init__(name,position,color,img_size)
  
  def calc_all_moves(self) -> List[Tuple]:
    
    row, col = self.position
    # all the possible directions
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  
    return directions
    #return the rows and cols +i multiplied by their directions to get all possible directions
    return [(row + i * dr, col + i * dc) for dr, dc in directions for i in range(1, 9) if 0 <= row + i * dr <= 7 and 0 <= col + i * dc <= 7]


      
  def move(self) -> None:
    print("d")
    