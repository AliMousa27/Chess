from .piece import Piece
from .piece_color import Piece_Color
from typing import List, Tuple

class Pawn(Piece):
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int) -> None:
    super().__init__(name,position,color,img_size)
    self.has_stepped=False
  
  def calc_all_moves(self) -> List[Tuple]:
    #for a pawn, we will include the en passant moves and filter them out in game logic
    row,col = self.position
    directions = [(1,1),(-1,-1),(1,0),(-1,0),(-1,1),(1,-1)]
    moves=[]
    
    if not self.has_stepped:
      new_row = row + 2 if self.color == Piece_Color.BLACK else row-2
      moves.append((new_row,col))
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
        
        
  def filter_pawn_moves(self,board:List[List],is_pinned,check_for_pins):
        row,col = self.position
        moves=[]
        for new_row,new_col in self.calc_all_moves():
            if check_for_pins and is_pinned(self,(new_row,new_col)):
                continue
            destination_square = board[new_row][new_col]
            #add en passant move if the square is occupied by an enemy pawn thats not on the same column as the pawn
            if new_col != col and destination_square.occupant and destination_square.occupant.color != self.color:
                moves.append((new_row,new_col))
            #linear vertical moves checks. Check first if the columns is the same and that the destiuon is empty
            elif destination_square.occupant is None and new_col == col:
                # then check if its moving 2 squares, then check if the square infront of it is empty
                if abs(new_row - row) == 2:
                    row_direction = 1 if self.color == Piece_Color.BLACK else -1
                    if board[row + row_direction][col].occupant is None:
                        moves.append((new_row,new_col))
                else:
                    moves.append((new_row,new_col))
        return moves
