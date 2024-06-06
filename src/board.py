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
    self.board=[[None]*8 for _ in range(8)]
    self.board: List[List[Square]] = self.setup()
    
  def setup(self):
    #each key is just a col and the value is a pices class over an object. This is done so we construct concrete objects with specifics colors 
    INITAL_PIECES_MAP= {0: Rook, 1: Knight, 2: Bishop, 3: Queen, 4: King, 5:Bishop, 6: Knight, 7:Rook}
    for row in range(8):
      for col in range(8):
        
        if row == 0:
          piece_class: Type[Piece] = INITAL_PIECES_MAP[col]
          piece = piece_class(piece_class.__name__, (row, col), Piece_Color.BLACK)
          self.board[row][col]=Square(row,col,True,piece)
          
        elif row == 1:
          piece = Pawn("Pawn",(row,col),Piece_Color.BLACK)
          self.board[row][col]=Square(row,col,True,piece)
          
        elif row == 6:
          piece = Pawn("Pawn",(row,col),Piece_Color.WHITE)
          self.board[row][col]=Square(row,col,True,piece)
          
        elif row == 7:
          piece_class: Type[Piece] = INITAL_PIECES_MAP[col]
          piece = piece_class(piece_class.__name__, (row, col), Piece_Color.WHITE)
          self.board[row][col]=Square(row,col,True,piece)
        else:
          self.board[row][col]=Square(row,col,False,None)
    return self.board
  def print_board(self):
      for row in self.board:
        for square in row:
          print(square, end=' ')
        print()

b = Board()
b.print_board()