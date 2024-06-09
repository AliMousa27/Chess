
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
        self.selected_piece = None  # Add this line

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
                        self.handle_click(square)
            self.board.clock.tick(60)
        pygame.quit()

    def handle_click(self, square_clicked: Square): 
      if square_clicked.occupant:
          self.selected_piece = square_clicked.occupant
          self.legal_moves = self.filter_moves(self.selected_piece.calc_all_moves(), self.selected_piece)
          self.board.highlight_moves(self.legal_moves, self.highlighted_moves)
      elif self.selected_piece:
          if (square_clicked.row, square_clicked.col) in self.legal_moves:
              self.move_piece(self.selected_piece, square_clicked)
              self.board.animate_move(self.selected_piece, square_clicked)
          self.board.restore_colors(self.highlighted_moves)
          self.selected_piece = None
          self.legal_moves = []

    def move_piece(self, piece: Piece, destination: Square):  # Add this method
        # Remove piece from current square
        current_square: Square = self.board.board[piece.position[0]][piece.position[1]]
        current_square.occupant = None
        current_square.is_occupied = False

        # Add piece to destination square
        destination.occupant = piece
        destination.is_occupied = True

        # Update piece's position
        piece.position = (destination.row, destination.col)

    def filter_moves(self, all_moves, piece: Piece):  # Add this method
        valid_moves = []
        for row, col in all_moves:
            square: Square = self.board.board[row][col]
            if not square.occupant or square.occupant.color != piece.color:
                valid_moves.append((row, col))
        return valid_moves
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()