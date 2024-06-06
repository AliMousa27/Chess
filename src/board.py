from pieces import *
from square import Square
from typing import List, Type
from pieces.piece_color import Piece_Color
from pieces.piece import Piece
from pieces.knight import Knight
from pieces.rook import Rook
from pieces.king import King
from pieces.queen import Queen
from pieces.bishop import Bishop
from pieces.pawn import Pawn

class Board():
  def __init__(self) -> None:
    self.board: List[List[Square]] = self.setup()
    
  def setup(self):
    #each key is just a col and the value is a pices class over an object. This is done so we construct concrete objects with specifics colors 
    INITAL_PIECES_MAP= {1: Rook, 2: Knight, 3: Bishop, 4: Queen, 5: King, 6:Bishop, 7: Knight, 8:Rook}
    for row in range(1,9):
      for col in range(1,9):
        
        if row == 1:
          piece_class: Type[Piece] = INITAL_PIECES_MAP[col]
          piece = piece_class(piece_class.__name__ (row, col), Piece_Color.BLACK)
          self.board[row][col]=Square(row,col,True,piece)
          
        elif row == 2:
          piece = Pawn("Pawn",(row,col),Piece_Color.BLACK)
          self.board[row][col]=Square(row,col,True,piece)
          
        elif row == 7:
          piece = Pawn("Pawn",(row,col),Piece_Color.WHITE)
          self.board[row][col]=Square(row,col,True,piece)
          
        elif row == 8:
          piece_class: Type[Piece] = INITAL_PIECES_MAP[col]
          piece = piece_class(piece_class.__name__ (row, col), Piece_Color.WHITE)
          self.board[row][col]=Square(row,col,True,piece)
        else:
          self.board[row][col]=Square(row,col,False,None)
          

b = Board()