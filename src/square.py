from pieces.piece import Piece
from pieces.piece_color import Piece_Color
from pygame import Rect
from pygame import Color
class Square():
  """
  Args:
    row: the row of the square
    col: the column of the square relative to the 2d array
    is_occupied a boolean to idnicate if theres a piece in the square
  
  """
  def __init__(self,row:int,col:int, is_occupied:bool, Occupant: Piece, size:int, color:Color) -> None:
    self.row=row
    self.col=col
    self.is_occupied=is_occupied
    self.occupant = Occupant
    self.color=color
    self.rect = Rect(size*col,size*row,size,size)
  
  #string representation to be printed in the board
  def __str__(self):

     if self.is_occupied:
       #for now return the first letter of the name jsut to be printed and see board layout
       color_str = "B" if self.occupant.color == Piece_Color.BLACK else "W"
       return f"{self.occupant.name[0]}_{color_str}"
     #return a random letter just to see empty squares
     else: return "X"
    

