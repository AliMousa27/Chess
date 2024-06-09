import pygame
from typing import List, Type, Dict, Tuple
from square import Square
from pieces.piece_color import Piece_Color
from pieces.piece import Piece
from pieces.knight import Knight
from pieces.rook import Rook
from pieces.king import King
from pieces.queen import Queen
from pieces.bishop import Bishop
from pieces.pawn import Pawn

#constants
SCREEN_SIZE = 400
SIZE= SCREEN_SIZE//8
GREEN_COLOR = 118,150,86,255
WHITE_COLOR = 238,238,210,255

class Board():
  def __init__(self) -> None:
    pygame.init()
    self.board: List[List[Square]]= [[None]*8 for _ in range(8)]
    self.screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
    self.clock = pygame.time.Clock()
    self.board: List[List[Square]] = self.setup()

  def draw_square(self, row: int, col: int):
    square = self.board[row][col]
    pygame.draw.rect(self.screen, square.color, square.rect)
    if square.is_occupied:
      self.screen.blit(square.occupant.image, (SIZE * col, SIZE * row))

  def setup(self):
    INITAL_PIECES_MAP: Dict[int,Type[Piece]] = {0: Rook, 1: Knight, 2: Bishop, 3: Queen, 4: King, 5:Bishop, 6: Knight, 7:Rook}
    for row in range(8):
      for col in range(8):
        curr_square_color = pygame.Color(GREEN_COLOR) if (row+col)%2==0 else pygame.Color(WHITE_COLOR)
        self.board[row][col]=Square(row,col,False,None,SIZE,curr_square_color)
        self.draw_square(row, col)

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

  def load_important_pieces(self,color:Piece_Color,row:int,col:int,pieces_class_map: Dict[int,Type[Piece]]):
    piece_class: Type[Piece] = pieces_class_map[col]
    piece = piece_class(piece_class.__name__, (row, col), color,SIZE)
    self.board[row][col].occupant= piece
    self.board[row][col].is_occupied=True
    self.draw_square(row, col)

  def load_pawn(self,color:Piece_Color,row:int,col:int):
    piece = Pawn("Pawn",(row,col),color,SIZE)
    self.board[row][col].occupant= piece
    self.board[row][col].is_occupied=True
    self.draw_square(row, col)

  def highlight_moves(self,possible_moves: List[Tuple],highlighted_squares):
    self.restore_colors(highlighted_squares)
    for row,col in possible_moves:
        original_color = pygame.Color(GREEN_COLOR) if (row+col)%2==0 else pygame.Color(WHITE_COLOR)
        highlighted_squares.append(((row, col), original_color))
        self.board[row][col].color = pygame.Color("red")
        self.draw_square(row, col)
    pygame.display.flip()

  def animate_move(self, piece: Piece, destination: Square):
    self.draw_board()
    self.screen.blit(piece.image, (SIZE * destination.col, SIZE * destination.row))
    pygame.display.flip()

  def draw_board(self):
    for row in range(8):
      for col in range(8):
        self.draw_square(row, col)
    pygame.display.flip()

  def restore_colors(self,highlighted_squares):
    for (row, col), color in highlighted_squares:
      self.board[row][col].color = color
      self.draw_square(row, col)
    pygame.display.flip()
    highlighted_squares = []