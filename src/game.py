
from board import Board
import pygame
from square import Square
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
            self.board.clock.tick(60)
        pygame.quit()
        
    def calc_all_moves_for_click(self, square_clicked:Square):
        if square_clicked.occupant:
          print("trynna highlight all koves")
          self.board.highlight_moves(square_clicked.occupant.calc_all_moves())
        else:
          print("tryna restore colors")
          self.board.restore_colors()
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()