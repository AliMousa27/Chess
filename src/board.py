import pygame
from pygame import Color
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
#the size of each square and piece. Can be used to move pieces across the screen by sing size*col index to move it horizontally for example
SIZE= SCREEN_SIZE//8
#square colors
GREEN_COLOR = 118,150,86,255
WHITE_COLOR = 238,238,210,255

class Board():
  def __init__(self) -> None:
    pygame.init()
    self.board: List[List[Square]]= [[None]*8 for _ in range(8)]
    self.screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
    self.clock = pygame.time.Clock()
    self.board: List[List[Square]] = self.setup()

  """
  Method to setup board at and draw the pieces
  Return:
    Returns a 2D list of squares that is the board itself
  """
  def setup(self) -> List[List[Square]]:
    #the key is the index of the column and the value is a piece concrete class so we can construct an object from
    INITAL_PIECES_MAP: Dict[int,Type[Piece]] = {0: Rook, 1: Knight, 2: Bishop, 3: Queen, 4: King, 5:Bishop, 6: Knight, 7:Rook}

    for row in range(8):
      for col in range(8):
        #get the square color. We add row and col to get the alternating look
        curr_square_color: Color = pygame.Color(GREEN_COLOR) if (row+col)%2==0 else pygame.Color(WHITE_COLOR)
        #create the square with no occupant for now
        self.board[row][col]=Square(row,col,False,None,SIZE,curr_square_color)
        self.draw_square(row, col)
        #if we are in the top row then we make a black piece that is a non pawn piece
        if row == 0:
          self.load_important_pieces(Piece_Color.BLACK,row,col,INITAL_PIECES_MAP)
        #row 1 = load a black pawn
        elif row == 1:
          self.load_pawn(Piece_Color.BLACK,row,col)
          # row 6 = load an white pawn
        elif row == 6:
          self.load_pawn(Piece_Color.WHITE,row,col)
        #row 7 == final row = load a white important pieces
        elif row == 7:
          self.load_important_pieces(Piece_Color.WHITE,row,col,INITAL_PIECES_MAP)
    #reflect the changes
    pygame.display.flip()
    return self.board
  
  """
  Function to draw the square on the screen given a row and column
  Args:
    row: integer of the columns row
    col: integer that indicates where the square should be drawn horizontally
  Return: void as it only draws on the screen
  """
  def draw_square(self, row: int, col: int) -> None:
    square: Square = self.board[row][col]
    pygame.draw.rect(self.screen, square.color, square.rect)
    #if there is an image, draw the image on top of the square
    if square.is_occupied:
      self.screen.blit(square.occupant.image, (SIZE * col, SIZE * row))
      
    """
    Loads an important piece onto the chess board.

    Args:
      color (Piece_Color): The color of the piece.
      row (int): The row index of the square where the piece will be placed.
      col (int): The column index of the square where the piece will be placed.
      pieces_class_map (Dict[int, Type[Piece]]): A dictionary mapping column indices to piece classes.

    Returns:
      None
    """
  def load_important_pieces(self, color: Piece_Color, row: int, col: int, pieces_class_map: Dict[int, Type[Piece]]) -> None:
    piece_class: Type[Piece] = pieces_class_map[col]
    piece : Piece = piece_class(piece_class.__name__, (row, col), color, SIZE)
    self.board[row][col].occupant = piece
    self.board[row][col].is_occupied = True
    self.draw_square(row, col)

    """
    Loads an pawn of a given color onto the chess board.

    Args:
      color (Piece_Color): The color of the piece.
      row (int): The row index of the square where the pawn will be placed.
      col (int): The column index of the square where the pawn will be placed.
      
    Returns:
      None
    """
  def load_pawn(self,color:Piece_Color,row:int,col:int) -> None:
    piece = Pawn("Pawn",(row,col),color,SIZE)
    self.board[row][col].occupant= piece
    self.board[row][col].is_occupied=True
    self.draw_square(row, col)
    
    """
    Highlights the possible moves on the chessboard.

    Args:
      possible_moves (List[Tuple]): A list of tuples where first element in the tuple is the row and the second is column representing the indices where a piece can move.
      highlighted_squares (List[Tuple[Tuple[int, int], Color]]): A list of tuples representing the squares to highlight and their original colors.

    Returns:
      None
    """
  def highlight_moves(self, possible_moves: List[Tuple], highlighted_squares: List[Tuple[Tuple[int, int], Color]]) -> None:
    # clear the previous highlighted squares
    self.restore_colors(highlighted_squares)
    for row, col in possible_moves:
      original_color = pygame.Color(GREEN_COLOR) if (row + col) % 2 == 0 else pygame.Color(WHITE_COLOR)
      # add the square to keep track of it
      highlighted_squares.append(((row, col), original_color))
      #highlight with red color
      self.board[row][col].color = pygame.Color("red")
      #draw the square
      self.draw_square(row, col)
    pygame.display.flip()
  """
  Animates the movement of a piece on the chessboard.
  Args:
    piece (Piece): The piece to move.
    destination (Square): The square to move the piece to.
  Returns:
    None
  """
  def animate_move(self, piece: Piece, destination: Square) -> None:
    self.draw_board()
    self.screen.blit(piece.image, (SIZE * destination.col, SIZE * destination.row))
    pygame.display.flip()
  """
  Function to draw the board from scratch on the screen
  Return: void as it only draws on the screen
  """
  def draw_board(self) -> None:
    for row in range(8):
      for col in range(8):
        self.draw_square(row, col)
    pygame.display.flip()
  """
  Function to restor the colors of the previously highlighted squares
  Args
    highlighted_squares: List of tuples where the first element is a tuple of row and column and the second element is the color of the square
  """
  def restore_colors(self,highlighted_squares : List[Tuple[Tuple[int, int], Color]]) -> None:
    #for each square with its row column and orginal color
    for (row, col), color in highlighted_squares:
      #get the square and set the color to the original color
      self.board[row][col].color = color
      #redraw the square
      self.draw_square(row, col)
    pygame.display.flip()
    highlighted_squares = []