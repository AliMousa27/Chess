import chess
import numpy as np
import torch
import torch.nn as nn
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
class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2,padding=0)
        
        self.relu = nn.ReLU()
        self.lin1= nn.Linear(256*1*1, 256)
        #64x64 output to represent the all possible from and to squares
        self.lin2 = nn.Linear(256, 64*64)
        
    def forward(self, x:torch.Tensor):
        # Forward pass through convolutional layers with batch normalization and ReLU
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # Flatten the output to pass it to the fully connected layers
        x = x.view(-1, 256*1*1)
        # Forward pass through fully connected layers with ReLU and Dropout
        x = self.relu(self.lin1(x)) 
        x = self.lin2(x)
        
        return x
    
def encode_move(move: chess.Move) -> np.array:
    print(f"the from index is {move.from_square} and the to index is {move.to_square}")
    return np.array([1 if i == move.from_square * 64 + move.to_square else 0 for i in range(64*64)])

def decode_move(encoded_move: np.array) -> chess.Move:
    index = np.argmax(encoded_move)
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

model = ChessModel()
matrix_test = board_to_tensor(board)
result = model(matrix_test.unsqueeze(0))
print(f"The shape is {result.shape} and the result is {result}")

move = chess.Move.from_uci("e2e4")
encoded_move = encode_move(move)
decoded_move = decode_move(encoded_move)

print(f"the decoded move is {decoded_move}")