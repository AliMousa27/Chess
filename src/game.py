from typing import List, Tuple
from board import Board
import pygame
from pieces import pawn
from pieces.pawn import Pawn
from pieces.piece import Piece
from pieces.knight import Knight
from pieces.rook import Rook
from square import Square
class Game:
    def __init__(self):
        self.board = Board()
        self.highlighted_moves=[]
        self.selected_piece = None  

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

    def move_piece(self, piece: Piece, destination: Square):
        # Remove piece from current square
        current_square: Square = self.board.board[piece.position[0]][piece.position[1]]
        current_square.occupant = None
        current_square.is_occupied = False

        # Add piece to destination square
        destination.occupant = piece
        destination.is_occupied = True

        # Update piece's position
        piece.position = (destination.row, destination.col)
        if isinstance(piece,Pawn): piece.has_stepped = True

    def filter_moves(self, all_moves, piece: Piece): 
        valid_moves = []

        if isinstance(piece,Knight):
            for row, col in all_moves:
                    square: Square = self.board.board[row][col]
                    if not square.occupant or square.occupant.color != piece.color:
                        valid_moves.append((row, col))
        elif isinstance(piece,Pawn):
            return self.filter_pawn_moves(piece,all_moves)
            
        else: return self.filter_linear_moves(piece)
                
                
        return valid_moves
    
    def filter_pawn_moves(self,pawn:Pawn,all_moves):
        col =pawn.position[1]
        moves=[]
        for new_row,new_col in all_moves:
            destination_square : Square = self.board.board[new_row][new_col]
            #add en passant move if the square is occupied by an enemy pawn thats not on the same column as the pawn
            if new_col != col and destination_square.occupant and destination_square.occupant.color != pawn.color:
                moves.append((new_row,new_col))
            elif destination_square.occupant is None and new_col == col:
                print(f"second if added {new_row},{new_col}")
                moves.append((new_row,new_col))
        return moves
        
    def filter_linear_moves(self, piece: Piece):
        #row col direct
        row,col = piece.position
        directions: Tuple[int,int] = piece.calc_all_moves()
        skip_direction = False
        moves = []
        for dr,dc in directions:
            
            for i in range(1,9):
                new_row = row + i * dr
                new_col = col + i * dc
                if not 0<=new_row<= 7 or not 0<=new_col<=7:
                    continue
                destination_square : Square = self.board.board[new_row][new_col]
                #if its occupied with the same color then skip that direction 
                if destination_square.occupant and destination_square.occupant.color == piece.color:
                    skip_direction = True
                    break
                elif destination_square.occupant and destination_square.occupant.color != piece.color:
                    moves.append((new_row,new_col))
                    skip_direction = True
                    break
                elif not destination_square.occupant:
                    moves.append((new_row,new_col))
            if skip_direction:
                skip_direction = False
                continue
                
        return moves
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()