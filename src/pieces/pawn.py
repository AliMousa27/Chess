from .piece import Piece
from .piece_color import Piece_Color
from typing import List, Tuple

class Pawn(Piece):
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int) -> None:
    super().__init__(name,position,color,img_size)
  
  def calc_all_moves(self) -> List[Tuple]:
    #for a pawn, we will include the en passant moves and filter them out in game logic
    row,col = self.position
    directions = [(1,1),(-1,-1),(1,0),(-1,0),(-1,1),(1,-1)]
    moves=[]
    for dr, dc in directions:
      #calculate the new squares
      new_row,new_col = dr+row,dc+col
      in_bounds= 0 <= new_row <= 7 and 0 <= new_col<= 7
      black_valid_move = in_bounds and new_row>row
      white_valid_move = in_bounds and new_row<row
      if self.color==Piece_Color.WHITE and white_valid_move:
        moves.append((new_row,new_col))
      elif black_valid_move and self.color==Piece_Color.BLACK:
        moves.append((new_row,new_col))
    return moves
        
        
        
      
  def move(self) -> None:
    print("d")