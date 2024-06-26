import chess
import numpy as np
import torch
import torch.nn as nn
import chess.pgn
from torch import optim
class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2,padding=0)
        
        self.relu = nn.ReLU()
        self.lin1= nn.Linear(256*1*1, 256)
        #64x64 output to represent the all possible from and to squares
        self.lin2 = nn.Linear(256, 64*64)
        
    def forward(self, x:torch.Tensor):
        # Forward pass through convolutional layers with batch normalization and ReLU
        x = self.pool(self.relu((self.conv1(x))))
        x = self.pool(self.relu((self.conv2(x))))
        x = self.pool(self.relu((self.conv3(x))))
        # Flatten the output to pass it to the fully connected layers
        x = x.view(-1, 256*1*1)
        # Forward pass through fully connected layers with ReLU and Dropout
        x = self.relu(self.lin1(x)) 
        x=self.dropout(x)
        # Forward pass through the output layer with sigmoid activation function to get probabilities
        x = torch.sigmoid(self.lin2(x))
        
        return x
    
    def board_to_tensor(self,board: chess.Board) -> torch.Tensor:
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

def encode_move(move: chess.Move) -> np.array:
    return np.array([1 if i == move.from_square * 64 + move.to_square else 0 for i in range(64*64)])

def decode_move(encoded_move: np.array,board: chess.Board) -> chess.Move:
    #create a list of 64*64 zeros to represent all possible moves
    encoded_legal_moves = [0 for _ in range(64*64)]
    #iterate over all the legal moves and set the index of the move to 1 to show all possible LEGAL MOVES
    for move in board.legal_moves:
        encoded_legal_moves[move.from_square*64 + move.to_square] = 1
    
    #set the encoded move to 0 if it is not a legal move
    encoded_move = np.multiply(encoded_move, encoded_legal_moves)
    
    #get the index of the max value in the encoded move and 
    #use divmod to get the from and to square to reverse the encoding
    from_square,to_square = divmod(np.argmax(encoded_move), 64)
    
    if all(encoded_move == 0):
        print(f"ajkefbeabfuewkfbhoewrjkfhleOIWEUBIK")
        return None
    
    return chess.Move(from_square, to_square)


# Assuming the rest of the code and necessary imports are already in place
def load_games(path,limit=500):
        games = []
        with open(path) as pgn_file:
            for _ in range(limit):
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        return games

def train(model, games_dataset, criterion, optimizer, num_epochs, device):
    model.to(device)  # Move model to device
    for epoch in range(num_epochs):
        for game in games_dataset:
            board = chess.Board()
            for move in game.mainline_moves():
                board_tensor = model.board_to_tensor(board).unsqueeze(0).to(device)  

                target_vector = torch.tensor(encode_move(move), dtype=torch.float32).unsqueeze(0).to(device) 
                optimizer.zero_grad()

                output = model(board_tensor)

                loss = criterion(output, target_vector)

                loss.backward()
                optimizer.step()
                    

                print(f"Epoch {epoch}, Loss: {loss.item()}")

                board.push(move)



path = __file__.replace("AI.py","data.pgn")

model = ChessModel()
#binary cross entropy loss
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
games_dataset=load_games(path)
train( model,games_dataset, criterion, optimizer, num_epochs=3, device=device)

test_game=games_dataset[0]
board = chess.Board()
for move in test_game.mainline_moves():
    board_tensor = model.board_to_tensor(board).unsqueeze(0).to(device)
    output = model(board_tensor)
    decoded_move = decode_move(output.cpu().detach().numpy()[0],board)
    print(f"Predicted move: {decoded_move}, Actual move: {move}")
    board.push(move)


