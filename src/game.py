from pieces.piece_color import Piece_Color
from board import Board
import pygame
from square import Square
from pieces.bishop import Bishop
from pprint import pprint
class Game:
    def __init__(self):
        self.board = Board()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                  x, y = pygame.mouse.get_pos()
                  for row in self.board.board:
                    for square in row:
                      if square.rect.collidepoint(x,y):
                        self.calc_all_moves_for_click(square)
                        #now that a square is clicked we need all the moves
                        #print(f"The square that was clikced has row {square.row+1}, {square.col+1}")
            self.board.clock.tick(60)
        pygame.quit()
        
    def calc_all_moves_for_click(self, square_clicked:Square):
      if square_clicked.occupant:
        color_str = "white" if square_clicked.occupant.color==Piece_Color.WHITE else "black"
        print(f"U clicked a {color_str} {square_clicked.occupant.name}")
        if isinstance(square_clicked.occupant,Bishop):
          print(f"All possible moves for the bishop at {square_clicked.occupant.position}")
          pprint(square_clicked.occupant.calc_all_moves())
          
        
      else:print("No occupant dummy")
      
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()