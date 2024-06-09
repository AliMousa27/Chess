from ast import Tuple
from square import Square
from typing import List, Type, Dict
from pieces.piece_color import Piece_Color
from pieces.piece import Piece
from pieces.knight import Knight
from pieces.rook import Rook
from pieces.king import King
from pieces.queen import Queen
from pieces.bishop import Bishop
from pieces.pawn import Pawn
import pygame

#constants
SCREEN_SIZE = 400
SIZE= SCREEN_SIZE//8
GREEN_COLOR = 118,150,86,255
WHITE_COLOR = 238,238,210,255

#class that has 2 main responsibilites 1. draw and render the board 2. House the pieces on squares to keep track of them
class Board():
  def __init__(self) -> None:
    pygame.init()
    self.board: List[List[Square]]= [[None]*8 for _ in range(8)]
    self.screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
    self.clock = pygame.time.Clock()
    self.board: List[List[Square]] = self.setup()
    self.highlighted_squares = []
    
  def setup(self):
    #each key is just a col and the value is a pices class over an object. This is done so we construct concrete objects with specifics colors 
    INITAL_PIECES_MAP: Dict[int,Type[Piece]] = {0: Rook, 1: Knight, 2: Bishop, 3: Queen, 4: King, 5:Bishop, 6: Knight, 7:Rook}
    #draw squares
    for row in range(8):
      for col in range(8):
        self.board[row][col]=Square(row,col,False,None,SIZE)
        #adding row and col gives the alternating of squares effect
        if (row+col)%2==0: pygame.draw.rect(self.screen, pygame.Color(GREEN_COLOR), self.board[row][col].rect)
        else: pygame.draw.rect(self.screen, pygame.Color(WHITE_COLOR), self.board[row][col].rect)
        
    for row in range(8):
      for col in range(8):
        if row == 0:
          self.load_important_pieces(Piece_Color.BLACK,row,col,INITAL_PIECES_MAP)
          
        elif row == 1:
          self.load_pawn(Piece_Color.BLACK,row,col)
          
        elif row == 6:
          self.load_pawn(Piece_Color.WHITE,row,col)
        elif row == 7:
          self.load_important_pieces(Piece_Color.WHITE,row,col,INITAL_PIECES_MAP)
          
    pygame.display.flip()
    return self.board
  
  """
  Function to load a given pawn on a given row and col
  Args:
    color: The gien color to a pawn 
    row: the row in the board
    col: the column of the board
    pieces_class_map: a map where the value key is the column where the important piece is and the value is a class refrence.
    -> Used for modularity purposes to load any given important piece quickly. The map is defined in the setup function
  """
  def load_important_pieces(self,color:Piece_Color,row:int,col:int,pieces_class_map: Dict[int,Type[Piece]]):
    #get the piece class by giving the column
    piece_class: Type[Piece] = pieces_class_map[col]
    #construct a piece
    piece = piece_class(piece_class.__name__, (row, col), color,SIZE)
    #draw the image
    self.screen.blit(piece.image, (SIZE*col, SIZE*row))
    self.board[row][col].occupant= piece
    self.board[row][col].is_occupied=True
    
  """
  Function to load a given pawn on a given row and col
  Args:
    color: The gien color to a pawn 
    row: the row in the board
    col: the column of the board
  """
  def load_pawn(self,color:Piece_Color,row:int,col:int):
    piece = Pawn("Pawn",(row,col),color,SIZE)
    self.screen.blit(piece.image, (SIZE*col, SIZE*row))
    self.board[row][col].occupant= piece
    self.board[row][col].is_occupied=True
    
  def highlight_moves(self,possible_moves: List[Tuple]):
    self.restore_colors()
    for row,col in possible_moves:
      original_color = pygame.Color(GREEN_COLOR) if (row+col)%2==0 else pygame.Color(WHITE_COLOR)
      self.highlighted_squares.append(((row, col), original_color))

      pygame.draw.rect(self.screen, pygame.Color("red"), self.board[row][col].rect)
      if self.board[row][col].is_occupied:
            self.screen.blit(self.board[row][col].occupant.image, (SIZE*col, SIZE*row))
    pygame.display.flip()

  def restore_colors(self):
    for (row, col), color in self.highlighted_squares:
      pygame.draw.rect(self.screen, color, self.board[row][col].rect)
      if self.board[row][col].is_occupied:
            self.screen.blit(self.board[row][col].occupant.image, (SIZE*col, SIZE*row))
    pygame.display.flip()
    self.highlighted_squares = []