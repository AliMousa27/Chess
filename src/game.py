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
    
    def __init__(self) -> None:
        """
        Constructor to create a game object thats responsible for running the game
        """
        #initialize the board and using composition to access the board
        self.board = Board()
        #list to keep track of the highlighted moves in order to find out if the player clicks a square to move to
        self.highlighted_moves=[]
        #selcted piece to move is tied to the highluighted moves object. If a highlighted move is clicked 
        # and its inside the valid moves of the selected piece, then the piece will move      
        self.selected_piece:Piece = None  
        #2 lists to keep track of the pieces on the board
        self.white_pieces:List[Piece] = [square.occupant for row in self.board.board for square in row if square.occupant and square.occupant.color == Piece_Color.WHITE]
        self.black_pieces:List[Piece] = [square.occupant for row in self.board.board for square in row if square.occupant and square.occupant.color == Piece_Color.BLACK]
        #dictionary to keep track of the valid moves for each piece
        self.moves : Dict[Piece, List[Tuple[int,int]]] = {}
        #start with white
        self.white_turn= True
        #calcualte the moves for all the white pieces
        self.setup(Piece_Color.WHITE)
        

    def setup(self, color: Piece_Color)-> None:
        """
        Calculate the moves for all the pieces of a given color
        Args:
            color: the color of the pieces to calculate the moves for
        Return:
            None
        """
        list_to_use = self.white_pieces if color == Piece_Color.WHITE else self.black_pieces
        for piece in list_to_use:
            moves = self.get_moves( piece)
            self.moves[piece] = moves
    
    def check_mate(self, color: Piece_Color) -> bool:
        """
        Method to check if a given color is checkmated
        Args:
            color: the color to check if its in checkmate
        Return: 
            True if the color is checkmated, False otherwise
        """
        king = self.get_king(color)
        
        list_to_use = self.white_pieces if color == Piece_Color.WHITE else self.black_pieces
        #get all the valid moves for the pieces of the given color
        valid_moves: List[List[Tuple[int,int]]] = [move for piece in list_to_use for move in self.moves[piece]]
        
        #if the king is in check and there are no valid moves then the color is checkmated
        #note that a move woint be added if the piece can move to a square and the king is still in check therefore we check if theres any move at all
        return self.is_in_check(king) and len(valid_moves) == 0
  
    def is_in_check(self, king: King) -> bool:
        """
        Method to check if a given king is in check
        args:
            king: the king to check if its in check
        Return:
            True if the king is in check, False otherwise
        """
        enemy_pieces = self.black_pieces if king.color == Piece_Color.WHITE else self.white_pieces        
        for enemy_piece in enemy_pieces:
            #for each enemy piece, get all the valid moves and check if the king is in any of them
            valid_enemy_moves:List[Tuple[int,int]] = self.get_moves( enemy_piece,check_for_pins=False)
            if king.position in valid_enemy_moves:
                return True
        return False

    
    def get_king(self,color:Piece_Color) -> King:
        """
        Method to find the king of a given color
        Args:
            color: the color of the king to find
        Return:
            the king of the given color
        """
        for piece in self.white_pieces if color == Piece_Color.WHITE else self.black_pieces:
            
            if isinstance(piece,King):
                return piece
            
    def is_pinned(self,piece: Piece, move:Tuple[int,int])-> bool:
        """
        method to simualte a move and check if it causes a pin
        Args:
            piece: the piece to simulate the move
            move: the destination of the move
        Return:
            True if the move causes a pin, False otherwise
        """
        
        #given a piece simulate a move to see if it causes a check
        destination_row,destination_col = move
        destination_square = self.board.board[destination_row][destination_col]
        return self.move_piece(piece,destination_square,True)   
          
    def get_moves(self, piece: Piece,check_for_pins=True) -> List[Tuple[int,int]]:
        return piece.filter_moves(self.board.board,self.is_pinned,check_for_pins)

    def move_piece(self, piece: Piece, destination: Square, simulate=False) -> bool:
        
        """
        method to move a piece and update the board
        Args:
            piece: the piece to move
            destination: the square to move the piece to
            simulate: a boolean to indicate if the move is a simulation or not to find if the moves causes a check
        Return:
            True if the move causes a check, False otherwise or nothing if its not a simulation
        """
        
        #castling move TODO
        
        
        # Save the current state
        current_square: Square = self.board.board[piece.position[0]][piece.position[1]]
        destination_square: Square = self.board.board[destination.row][destination.col]
        original_occupant = destination_square.occupant
        
        # Move the piece
        current_square.occupant = None
        current_square.is_occupied = False
        destination_square.occupant = piece
        destination_square.is_occupied = True
        piece.position = (destination.row, destination.col)
        # remove from the list if the piece is captured
        if original_occupant and original_occupant.color == Piece_Color.WHITE and not isinstance(original_occupant,King):
            self.white_pieces.remove(original_occupant)
        elif original_occupant and original_occupant.color == Piece_Color.BLACK and not isinstance(original_occupant,King):
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
            
    def promote(self,piece:Piece, destination:Square) -> None:
        """
        Method to promote a pawn to a given piece
        Args:
            piece: the piece to promote
            destination: the square to promote the piece to
        Return:
            None
        """
        #ask the user for the piece to promote to
        choice=-1
        while not (0 < choice < 5):
            #input validation
            try:
                choice = int(input("Choose piece to promote to:\n1.Queen\n2.Rook\n3.Bishop\n4.Knight"))
            except Exception as e:
                print("Invalid choice. Please put number between 1-4")
        #dictionary to map the choice to the piece
        CHOICES_MAP: Dict[int,Type[Piece]] = {1: Queen, 2: Rook, 3: Bishop, 4:Knight }
        #get the piece class from the dictionary and create a new piece
        piece_class: Type[Piece] = CHOICES_MAP[choice]
        new_piece : Piece = piece_class(piece_class.__name__, piece.position, piece.color, 50)
        destination.occupant = new_piece
        list_to_use = self.white_pieces if piece.color == Piece_Color.WHITE else self.black_pieces
        #remove the old piece and add the new piece
        list_to_use.remove(piece)
        list_to_use.append(new_piece)

        
    def run(self) -> None:
        """
        Method to run the game and handle the events
        """
        #standard pygame loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                  x, y = pygame.mouse.get_pos()
                  #if we clicked a square then handle the click
                  for row in self.board.board:
                    for square in row:
                      if square.rect.collidepoint(x,y):
                        self.handle_click(square)
            self.board.clock.tick(60)
        pygame.quit()

    def handle_click(self, square_clicked: Square): 
        """
        Method to handle the click event. Move a piece or highlight the moves of a piece or do nothing
        Args:
            square_clicked: the square that was clicked
        Return:
            None
        """
        #if we clicked on a square and there is a selected piece then move the piece
        if self.selected_piece:
                #check if the clicked square is a valid move. The second arg in the get 
                # method is the default value if the key is not found so we return an empty list to signify false
            if (square_clicked.row, square_clicked.col) in self.moves.get(self.selected_piece, []):
                #move the piece and animate the move
                self.move_piece(self.selected_piece, square_clicked)
                self.board.animate_move(self.selected_piece, square_clicked)
                #change the turn and calculate the moves for the other color
                self.white_turn = not self.white_turn
                color = Piece_Color.WHITE if self.white_turn else Piece_Color.BLACK
                self.setup(color)
                #check if the other color is in check
                color_being_attacked = Piece_Color.BLACK if self.white_turn else Piece_Color.WHITE
                is_checked = self.check_mate(color_being_attacked)
                if is_checked:
                    print(f"the color {self.selected_piece.color.name} has been checkmated")
                    exit(0)
            #restore the colors of the highlighted moves to the original color
            self.board.restore_colors(self.highlighted_moves)
            self.selected_piece = None
            
            
        #else if we clicked on a square that has a piece and its the same color as the turn then highlight the moves
        elif square_clicked.occupant and square_clicked.occupant.color == Piece_Color.WHITE and self.white_turn:
            self.selected_piece = square_clicked.occupant
            self.board.highlight_moves(self.moves[self.selected_piece], self.highlighted_moves)
                
        elif square_clicked.occupant and square_clicked.occupant.color == Piece_Color.BLACK and not self.white_turn:
            self.selected_piece = square_clicked.occupant
            self.board.highlight_moves(self.moves[self.selected_piece], self.highlighted_moves)

        
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
    