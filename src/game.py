from typing import List, Tuple, Dict, Type
from board import Board
import pygame
from pieces.bishop import Bishop
from pieces.knight import Knight
from pieces.pawn import Pawn
from pieces.piece import Piece
from pieces.king import King
from pieces.queen import Queen
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
        self.moves : Dict[Piece, List[Tuple[int,int]]] = {}
        self.white_turn= True
        self.setup()
        

    def setup(self):
        for piece in self.white_pieces + self.black_pieces:
            moves = self.get_moves( piece)
            self.moves[piece] = moves
    
    def check_mate(self, color: Piece_Color) -> bool:
        king = self.get_king(color)
        if color == Piece_Color.WHITE:
            valid_moves = [move for piece in self.white_pieces for move in self.moves[piece]]
        else:
            valid_moves = [move for piece in self.black_pieces for move in self.moves[piece]]
        
        return self.is_in_check(king) and len(valid_moves) == 0
        #OLD IMPLEMENTATION
        '''king = self.get_king(color)
        valid_moves = []
        
        for piece in self.white_pieces if color == Piece_Color.WHITE else self.black_pieces:
            valid_moves.extend(self.get_moves( piece))
        return self.is_in_check(king) and len(valid_moves) == 0'''

    
    def get_king(self,color:Piece_Color):
        for piece in self.white_pieces if color == Piece_Color.WHITE else self.black_pieces:
            
            if isinstance(piece,King):
                return piece
            
    def is_in_check(self, king: King) -> bool:

        enemy_pieces = self.black_pieces if king.color == Piece_Color.WHITE else self.white_pieces        
        for enemy_piece in enemy_pieces:
            valid_enemy_moves = self.get_moves( enemy_piece,check_for_pins=False)
            if king.position in valid_enemy_moves:
                return True
        return False
    
    def is_pinned(self,piece: Piece, move:Tuple[int,int]):
        #given a piece simulate a move to see if it causes a check
        destination_row,destination_col = move
        destination_square = self.board.board[destination_row][destination_col]
        return self.move_piece(piece,destination_square,True)   
          
    def get_moves(self, piece: Piece,check_for_pins=True):
        return piece.filter_moves(self.board.board,self.is_pinned,check_for_pins)

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
        # remove from the list
        if original_occupant and original_occupant.color == Piece_Color.WHITE:
            self.white_pieces.remove(original_occupant)
        elif original_occupant and original_occupant.color == Piece_Color.BLACK:
            self.black_pieces.remove(original_occupant)
            

        if simulate:
            # If it's a simulation, restore the original state and return the result
            is_check = self.is_in_check(self.get_king(piece.color))
            current_square.occupant = piece
            current_square.is_occupied = True
            destination_square.occupant = original_occupant
            destination_square.is_occupied = (original_occupant is not None)
            piece.position = (current_square.row, current_square.col)
            #add to the list
            if original_occupant and original_occupant.color == Piece_Color.WHITE:
                self.white_pieces.append(original_occupant)
            elif original_occupant and original_occupant.color == Piece_Color.BLACK:
                
                self.black_pieces.append(original_occupant)
            return is_check

        # If it's not a simulation, finalize the move
        if isinstance(piece,Pawn) or isinstance(piece,King) or isinstance(piece,Rook): piece.has_stepped = True
        #pawn promotion 
        if isinstance(piece, Pawn) and ((piece.color == Piece_Color.BLACK and piece.position[0] == 7) or (piece.color == Piece_Color.WHITE and piece.position[0] == 0)):
            self.promote(piece,destination)
            
    def promote(self,piece:Piece, destination:Square):
        choice=-1
        while not (0 < choice < 5):
            try:
                choice = int(input("Choose piece to promote to:\n1.Queen\n2.Rook\n3.Bishop\n4.Knight"))
            except Exception as e:
                print("Invalid choice. Please put number between 1-4")
                
        CHOICES_MAP: Dict[int,Type[Piece]] = {1: Queen, 2: Rook, 3: Bishop, 4:Knight }
        piece_class: Type[Piece] = CHOICES_MAP[choice]
        new_piece : Piece = piece_class(piece_class.__name__, piece.position, piece.color, 50)
        destination.occupant = new_piece
        
        if piece.color == Piece_Color.BLACK:
            self.black_pieces.remove(piece)
            self.black_pieces.append(new_piece)
        else:
            self.white_pieces.remove(piece)
            self.white_pieces.append(new_piece)
        
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
                #check if the clicked square is a valid move. The second arg in the get 
                # method is the default value if the key is not found so we return an empty list to signify false
                if (square_clicked.row, square_clicked.col) in self.moves.get(self.selected_piece, []):
                    self.move_piece(self.selected_piece, square_clicked)
                    self.board.animate_move(self.selected_piece, square_clicked)
                    color_being_attacked = Piece_Color.BLACK if self.white_turn else Piece_Color.WHITE
                    self.white_turn = not self.white_turn
                    self.setup()
                    is_checked = self.check_mate(color_being_attacked)
                    print(f"Is {color_being_attacked.name} in check: {is_checked}")
                    if is_checked:
                        print(f"the color {self.selected_piece.color.name} has been checkmated")
                        exit(0)
                    
                self.board.restore_colors(self.highlighted_moves)
                self.selected_piece = None
            
            
                
            elif square_clicked.occupant and square_clicked.occupant.color == Piece_Color.WHITE and self.white_turn:
                self.selected_piece = square_clicked.occupant
                self.board.highlight_moves(self.moves[self.selected_piece], self.highlighted_moves)
                
            elif square_clicked.occupant and square_clicked.occupant.color == Piece_Color.BLACK and not self.white_turn:
                self.selected_piece = square_clicked.occupant
                self.board.highlight_moves(self.moves[self.selected_piece], self.highlighted_moves)

    def print_board_state(self):
        white_pieces = 0
        black_pieces = 0
        for piece in self.white_pieces + self.black_pieces:
            
            
            if piece.color == Piece_Color.WHITE:
                white_pieces += 1
            else:
                black_pieces += 1
                print(f"Piece: {piece.name}, Color: {piece.color.name}, Position: ({piece.position[0]}, {piece.position[1]})")
        print(f"Total white pieces: {white_pieces}")
        print(f"Total black pieces: {black_pieces}")
        
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()