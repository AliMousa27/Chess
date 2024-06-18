from .piece import Piece
from .piece_color import Piece_Color
from typing import List, Tuple

class King(Piece):
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int) -> None:
    super().__init__(name,position,color,img_size)
    self.has_stepped=False
  
  def calc_all_moves(self) -> List[Tuple]:
    row,col = self.position
    directions = [(1,1),(1,-1),(-1,1),(-1,-1),(0,-1),(-1,0),(1,0),(0,1)]
    return [(dr+row,dc+col) for dr,dc in directions if 0 <= dr+row <= 7 and 0 <= dc+col <= 7]
      

  
  def filter_moves(self,board: List[List],is_pinned,check_for_pins:bool):
      moves = [(row,col) for row,col in self.calc_all_moves() if self.can_move_to_square(board,row,col,check_for_pins,is_pinned)] 
      #castling move
      if not self.has_stepped:
          rooks_row = 0 if self.color ==Piece_Color.BLACK else 7
          rook_has_stepped = board[rooks_row][7].occupant.has_stepped
          
          if self.can_swap_with_rook(board[rooks_row],rook_has_stepped): 
              moves.append((rooks_row,7))
      return moves
  
  def can_swap_with_rook(self,row,rook_has_stepped: bool) -> bool:
    #we dont check if the king has moved because game class does that
    can_castle = all(row[i].occupant is None for i in range(5,7))and not rook_has_stepped
    return can_castle
  
  
  def can_move_to_square(self,board:List[List],row:int,col:int,check_for_pins:bool,is_pinned) -> bool:
    return (not board[row][col].occupant or board[row][col].occupant.color != self.color) and (not check_for_pins or not is_pinned(self,(row,col)))