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
import pygame

SCREEN_SIZE = 800

class Board():
  def __init__(self) -> None:
    pygame.init()
    self.board=[[None]*8 for _ in range(8)]
    self.screen = pygame.display.set_mode((900, 900))
    self.clock = pygame.time.Clock()
    self.board: List[List[Square]] = self.setup()
    
  def setup(self):

    size = SCREEN_SIZE//8
    #each key is just a col and the value is a pices class over an object. This is done so we construct concrete objects with specifics colors 
    INITAL_PIECES_MAP= {0: Rook, 1: Knight, 2: Bishop, 3: Queen, 4: King, 5:Bishop, 6: Knight, 7:Rook}
    for row in range(8):
      for col in range(8):
        
        if (row+col)%2==0: pygame.draw.rect(self.screen, "black", [size*col,size*row,size,size])
        else: pygame.draw.rect(self.screen, "white", [size*col,size*row,size,size])
        
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
    pygame.display.flip()
    return self.board
  



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