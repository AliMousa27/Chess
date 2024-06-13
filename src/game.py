import copy
from re import S, T
from typing import List, Tuple
from board import Board
import pygame
from pieces.pawn import Pawn
from pieces.piece import Piece
from pieces.knight import Knight
from pieces.king import King
from pieces.rook import Rook
from pieces.piece_color import Piece_Color
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
      if self.selected_piece:
          if (square_clicked.row, square_clicked.col) in self.legal_moves:
              self.move_piece(self.selected_piece, square_clicked)
              self.board.animate_move(self.selected_piece, square_clicked)
              #print("is CHECKED",self.is_in_check(self.board.board[0][4].occupant))
          self.board.restore_colors(self.highlighted_moves)
          self.selected_piece = None
          self.legal_moves = []
          
          
      elif square_clicked.occupant:
          self.selected_piece = square_clicked.occupant
          self.legal_moves = self.filter_moves(self.selected_piece.calc_all_moves(), self.selected_piece, True)
          self.board.highlight_moves(self.legal_moves, self.highlighted_moves)
      
    def is_in_check(self, king: King) -> bool:
        enemy_color = Piece_Color.BLACK if king.color == Piece_Color.WHITE else Piece_Color.WHITE
        enemy_pieces: List[Piece] = [square.occupant for row in self.board.board for square in row if square.occupant and square.occupant.color == enemy_color]

        for enemy_piece in enemy_pieces:
            enemy_moves = enemy_piece.calc_all_moves()
            valid_enemy_moves = self.filter_moves(enemy_moves, enemy_piece)
            if king.position in valid_enemy_moves:
                return True

        return False
    

        

        
    
    def move_piece(self, piece: Piece, destination: Square,is_simulation=False):
        #check if its a castling move
        if isinstance(piece, King) and destination.occupant is not None and destination.occupant.color == piece.color and destination.col == 7:
            rook = self.board.board[destination.row][7].occupant
            destination = self.board.board[destination.row][6]
            self.move_piece(rook,self.board.board[destination.row][5])
            
        
        current_square: Square = self.board.board[piece.position[0]][piece.position[1]]
        current_square.occupant = None
        current_square.is_occupied = False

        temp = destination.occupant
        # Add piece to destination square
        
        destination.occupant = piece
        destination.is_occupied = True

        # Update piece's position
        piece.position = (destination.row, destination.col)
        if isinstance(piece,Pawn) or isinstance(piece,King) or isinstance(piece,Rook): piece.has_stepped = True

        return temp
        
        
    
    def filter_moves(self, all_moves, piece: Piece, check_for_pins: bool = False):
        valid_moves = []
        if isinstance(piece,Knight): 
            valid_moves= self.filter_knight_moves(piece,all_moves)
        elif isinstance(piece,King): valid_moves= self.filter_king_moves(piece,all_moves)
        elif isinstance(piece,Pawn): valid_moves= self.filter_pawn_moves(piece,all_moves)
        else: valid_moves= self.filter_linear_moves(piece)
        if check_for_pins:
            pass
            #if self.is_pinned(piece,valid_moves):
                #return []
        return valid_moves
                
    def filter_king_moves(self,piece:Piece,all_moves):
        moves = [(row,col) for row,col in all_moves if not self.board.board[row][col].occupant or self.board.board[row][col].occupant.color != piece.color] 
        #castling move
        if not piece.has_stepped:
            rooks_row = 0 if piece.color ==Piece_Color.BLACK else 7
            self.print_board_state()
            rook_has_stepped = self.board.board[rooks_row][7].occupant.has_stepped
            can_castle = piece.swap_with_rook(self.board.board[rooks_row],rook_has_stepped)
            if can_castle: 
                moves.append((rooks_row,7))
        
        return moves
            
    def filter_knight_moves(self,piece,all_moves):
        return [(row,col) for row,col in all_moves if not self.board.board[row][col].occupant or self.board.board[row][col].occupant.color != piece.color] 
    
    def filter_pawn_moves(self,pawn:Pawn,all_moves):
        row,col = pawn.position
        moves=[]
        for new_row,new_col in all_moves:
            destination_square : Square = self.board.board[new_row][new_col]
            #add en passant move if the square is occupied by an enemy pawn thats not on the same column as the pawn
            if new_col != col and destination_square.occupant and destination_square.occupant.color != pawn.color:
                moves.append((new_row,new_col))
            #linear vertical moves checks. Check first if the columns is the same and that the destiuon is empty
            elif destination_square.occupant is None and new_col == col:
                # then check if its moving 2 squares, then check if the square infront of it is empty
                if abs(new_row - row) == 2:
                    row_direction = 1 if pawn.color == Piece_Color.BLACK else -1
                    if self.board.board[row + row_direction][col].occupant is None:
                        moves.append((new_row,new_col))
                else:
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
    
    def print_board_state(self):
        white_pieces = 0
        black_pieces = 0
        for row in self.board.board:
            for square in row:
                if square.occupant:
                    piece = square.occupant
                    print(f"Piece: {piece.name}, Color: {piece.color.name}, Position: ({piece.position[0]}, {piece.position[1]})")
                    if piece.color == Piece_Color.WHITE:
                        white_pieces += 1
                    else:
                        black_pieces += 1
        print(f"Total white pieces: {white_pieces}")
        print(f"Total black pieces: {black_pieces}")
        
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()