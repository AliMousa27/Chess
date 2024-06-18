from typing import List, Tuple
from board import Board
import pygame
from pieces.bishop import Bishop
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
        self.white_pieces = [square.occupant for row in self.board.board for square in row if square.occupant and square.occupant.color == Piece_Color.WHITE]
        self.black_pieces = [square.occupant for row in self.board.board for square in row if square.occupant and square.occupant.color == Piece_Color.BLACK]
        for i in range(len(self.white_pieces)):
            self.white_pieces[i].possible_moves=self.filter_moves(self.white_pieces[i].calc_all_moves(),self.white_pieces[i])
            self.black_pieces[i].possible_moves=self.filter_moves(self.black_pieces[i].calc_all_moves(),self.black_pieces[i])   


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
              old_square = self.board.board[self.selected_piece.position[0]][self.selected_piece.position[1]]
              self.move_piece(self.selected_piece, square_clicked)
              
              for piece in self.find_affected_pieces( old_square,square_clicked):
                    piece.possible_moves = self.filter_moves(piece.calc_all_moves(),piece)
                    print(f"Piece: {piece.name}, Color: {piece.color.name}, Position: ({piece.position[0]}, {piece.position[1]}), Possible Moves: {piece.possible_moves}")
              self.board.animate_move(self.selected_piece, square_clicked)
          self.board.restore_colors(self.highlighted_moves)
          self.selected_piece = None
          self.legal_moves = []
          
          
      elif square_clicked.occupant:
          self.selected_piece = square_clicked.occupant
          self.legal_moves = square_clicked.occupant.possible_moves
          #self.legal_moves = self.filter_moves(self.selected_piece.calc_all_moves(), self.selected_piece)
          self.board.highlight_moves(self.legal_moves, self.highlighted_moves)
    
    def find_affected_pieces(self, old_square:Square, new_square: Square):
        #this accounts for when a piece could move to a place but now it cant
        old_row,old_col = old_square.row,old_square.col
        new_row, new_col = new_square.row, new_square.col
        affected_pieces = []
        for piece in self.white_pieces + self.black_pieces:
            if ((new_row, new_col) in piece.possible_moves):
                affected_pieces.append(piece)
            #now we need to find if a piece could move in that path
            row, col = piece.position
            if isinstance(piece,Bishop) and (old_row,old_col) in [(row + i * dr, col + i * dc) for dr, dc in piece.directions for i in range(1, 9) if 0 <= row + i * dr <= 7 and 0 <= col + i * dc <= 7]:
                affected_pieces.append(piece)
        print(f"affected pieces {affected_pieces}")
        return affected_pieces
        
    
    def is_in_check(self, king: King) -> bool:
        enemy_pieces = self.black_pieces if king.color == Piece_Color.WHITE else self.white_pieces

        for enemy_piece in enemy_pieces:
            valid_enemy_moves = self.filter_moves(enemy_piece.calc_all_moves(), enemy_piece,check_for_pins=False)
            if king.position in valid_enemy_moves:
                return True

        return False

            

        
    
    def move_piece(self, piece: Piece, destination: Square, simulate=False):
        # Save the current state
        current_square: Square = self.board.board[piece.position[0]][piece.position[1]]
        destination_square: Square = self.board.board[destination.row][destination.col]
        original_occupant = destination_square.occupant

        current_square.occupant = None
        current_square.is_occupied = False
        destination_square.occupant = piece
        destination_square.is_occupied = True
        piece.position = (destination.row, destination.col)

        if simulate:
            # If it's a simulation, restore the original state and return the result
            is_check = self.is_in_check(self.get_king(piece.color))
            current_square.occupant = piece
            current_square.is_occupied = True
            destination_square.occupant = original_occupant
            destination_square.is_occupied = (original_occupant is not None)
            piece.position = (current_square.row, current_square.col)
            return is_check

        # If it's not a simulation, finalize the move
        if isinstance(piece, Pawn) or isinstance(piece, King) or isinstance(piece, Rook):
            piece.has_stepped = True

        
        

    def is_pinned(self,piece: Piece, move:Tuple[int,int]):
        #given a piece simulate a move to see if it causes a check
        destination_row,destination_col = move
        destination_square = self.board.board[destination_row][destination_col]
        return self.move_piece(piece,destination_square,True)
    
    def filter_moves(self, all_moves, piece: Piece,check_for_pins=True):
        if isinstance(piece,Knight): 
            return self.filter_knight_moves(piece,all_moves,check_for_pins)
        elif isinstance(piece,King): return self.filter_king_moves(piece,all_moves,check_for_pins)
        elif isinstance(piece,Pawn): return self.filter_pawn_moves(piece,all_moves,check_for_pins)
        else: return self.filter_linear_moves(piece,check_for_pins)

                
    def filter_king_moves(self,piece:Piece,all_moves,check_for_pins):
        moves = [(row,col) for row,col in all_moves if (not self.board.board[row][col].occupant or self.board.board[row][col].occupant.color != piece.color) and (not check_for_pins or not self.is_pinned(piece,(row,col)))] 
        #castling move
        if not piece.has_stepped:
            rooks_row = 0 if piece.color ==Piece_Color.BLACK else 7
            rook_has_stepped = self.board.board[rooks_row][7].occupant.has_stepped
            can_castle = piece.swap_with_rook(self.board.board[rooks_row],rook_has_stepped)
            if can_castle: 
                moves.append((rooks_row,7))
        
        return moves
            
    def filter_knight_moves(self,piece,all_moves,check_for_pins):
        return [(row,col) for row,col in all_moves if (not self.board.board[row][col].occupant or self.board.board[row][col].occupant.color != piece.color) and (not check_for_pins or not self.is_pinned(piece,(row,col)))] 
    
    def filter_pawn_moves(self,pawn:Pawn,all_moves,check_for_pins):
        row,col = pawn.position
        moves=[]
        for new_row,new_col in all_moves:
            if check_for_pins and self.is_pinned(pawn,(new_row,new_col)):
                continue
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
        
    def filter_linear_moves(self, piece: Piece,check_for_pins):
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
                if check_for_pins and self.is_pinned(piece,(new_row,new_col)):
                    continue
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
    
    def get_king(self,color:Piece_Color):
        for row in self.board.board:
            for square in row:
                if square.occupant and isinstance(square.occupant,King) and square.occupant.color == color:
                    return square.occupant
    
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