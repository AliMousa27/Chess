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

SCREEN_SIZE = 600

class Board():
  def __init__(self) -> None:
    pygame.init()
    self.board=[[None]*8 for _ in range(8)]
    self.screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
    self.clock = pygame.time.Clock()
    #attribute to hold the images
    self.piece_images  = self.load_piece_images()
    self.board: List[List[Square]] = self.setup()
    
    
  def setup(self):

    size = SCREEN_SIZE//8
    #each key is just a col and the value is a pices class over an object. This is done so we construct concrete objects with specifics colors 
    INITAL_PIECES_MAP= {0: Rook, 1: Knight, 2: Bishop, 3: Queen, 4: King, 5:Bishop, 6: Knight, 7:Rook}
    for row in range(8):
      for col in range(8):
        
        if (row+col)%2==0: pygame.draw.rect(self.screen, pygame.Color(118,150,86,255), [size*col,size*row,size,size])
        else: pygame.draw.rect(self.screen, pygame.Color(238,238,210,255), [size*col,size*row,size,size])
        
    for row in range(8):
      for col in range(8):
        
        
        if row == 0:
          piece_class: Type[Piece] = INITAL_PIECES_MAP[col]
          piece = piece_class(piece_class.__name__, (row, col), Piece_Color.BLACK)
          print(piece.name)
          self.board[row][col]=Square(row,col,True,piece)
          
          piece_image = self.piece_images[f'{piece_class.__name__.lower()}_black']
          self.screen.blit(piece_image, (size*col, size*row))
          
        elif row == 1:
          piece = Pawn("Pawn",(row,col),Piece_Color.BLACK)
          self.board[row][col]=Square(row,col,True,piece)
          piece_image = self.piece_images[f'{piece_class.__name__.lower()}_black']
          self.screen.blit(piece_image, (size*col, size*row))
        elif row == 6:
          piece = Pawn("Pawn",(row,col),Piece_Color.WHITE)
          self.board[row][col]=Square(row,col,True,piece)
          piece_image = self.piece_images[f'{piece_class.__name__.lower()}_white']
          self.screen.blit(piece_image, (size*col, size*row))
        elif row == 7:
          piece_class: Type[Piece] = INITAL_PIECES_MAP[col]
          piece = piece_class(piece_class.__name__, (row, col), Piece_Color.WHITE)
          self.board[row][col]=Square(row,col,True,piece)
          piece_image = self.piece_images[f'{piece_class.__name__.lower()}_white']
          self.screen.blit(piece_image, (size*col, size*row))
        else:
          self.board[row][col]=Square(row,col,False,None)
    pygame.display.flip()
    return self.board
  
  def load_piece_images(self):
      #the size of each image will be divided by 8 to get the scaled down size
      size = SCREEN_SIZE // 8
      piece_images = {}
      pieces = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
      colors = ['black', 'white']

      for piece in pieces:
          for color in colors:
              image = pygame.image.load(f'src/assets/{piece}_{color}.png')
              piece_images[f'{piece}_{color}'] = pygame.transform.scale(image, (size, size)) 

      return piece_images

class Game:
    def __init__(self):
        self.board = Board()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.board.clock.tick(60)
        pygame.quit()

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()