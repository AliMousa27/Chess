import chess
import numpy as np
import torch
import os
import chess.pgn

"""    
code to oad games. use later
def _load_games(self,path):
        games = []
        with open(path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        return games"""

    
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    # create 8x8x12 tensor (8x8 board, 12 planes(type of piece and color is the plane))
    #so 12 lists for each type of piece each list is a 2d 8x8 matrix
    board_matrix = np.zeros((8, 8, 12), dtype=np.int8)
    #create a dict to map the piece to the plane. This 
    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    #iterate over the board and fill the board_matrix
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            #get the plane for the piece
            plane = piece_to_plane[piece.symbol()]
            #map the square to the row and column using divmod  
            row, col = divmod(square, 8)
            #set the plane at that row and col to 1
            board_matrix[row, col, plane] = 1
    #return a tensor of the board_matrix and change 8 8 12 to 12 8 8 to represent pieces on the board
    return torch.tensor(board_matrix, dtype=torch.float32).permute(2, 0, 1) 

board = chess.Board()
print(board_to_tensor(board))
