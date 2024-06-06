from pieces.piece import Piece
from pieces.piece_color import Piece_Color

class Square():
  """
  Args:
    row: the row of the square
    col: the column of the square relative to the 2d array
    is_occupied a boolean to idnicate if theres a piece in the square
  
  """
  def __init__(self,row:int,col:int, is_occupied:bool, Occupant: Piece) -> None:
    self.row=row
    self.col=col
    self.is_occupied=is_occupied
    self.occupant = Occupant
    
  #change string representation to be printed in the board
  def __str__(self):
     if self.is_occupied:
       #for now return the first letter of the name jsut to be printed and see board layout
       return self.occupant.name[0]
     #return a random letter just to see empty squares
     else: return "X"
     
  def has_enemy_piece(self,color:Piece_Color):
    if self.occupant.color !=color: return True
    else: return False