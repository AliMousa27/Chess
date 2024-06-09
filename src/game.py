
from math import e
from typing import List
from board import Board
import pygame
from pieces.piece import Piece
from pieces.knight import Knight
from square import Square
class Game:
    def __init__(self):
        self.board = Board()
        self.highlighted_moves=[]

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
          all_moves=self.filter_moves(square_clicked.occupant.calc_all_moves(),square_clicked.occupant)
          self.board.highlight_moves(all_moves,self.highlighted_moves)
        else:
          self.board.restore_colors(self.highlighted_moves)
          
    def filter_moves(self,all_moves,piece:Piece):
      print(all_moves)
      #if its a knight we can just return all the moves that are not occupied by the same color 
      #as knight can jump over pieces
      if isinstance(piece,Knight):
        nice = []
        for row, col in all_moves:
            square: Square = self.board.board[row][col]
            if not square.occupant or square.occupant.color != piece.color:
                nice.append((row, col))
        return nice
      
      
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()