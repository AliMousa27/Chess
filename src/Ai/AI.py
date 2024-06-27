import chess
import numpy as np
import torch
import torch.nn as nn
import chess.pgn
from torch import optim
from torch.utils.data import Dataset

from Ai.data_handler import ChessDataset


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
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lin1 = nn.Linear(256, 256)
        self.lin2 = nn.Linear(256, 64*64)
        
    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(-1, 256)
        x = self.dropout(self.relu(self.lin1(x)))
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

def decode_move(encoded_move: np.array, board: chess.Board) -> chess.Move:
    # Generate a list of legal moves from the current board state
    legal_moves = list(board.legal_moves)

    # Create a list of zeros for all possible moves
    encoded_legal_moves = np.zeros(64*64)

    # Iterate over all legal moves and set the index of the move to 1
    for move in legal_moves:
        encoded_legal_moves[move.from_square * 64 + move.to_square] = 1

    # Multiply the encoded move with encoded_legal_moves to keep only legal moves by setting illegal moves to 0
    #then taking the most likely probable move the ai thinks is best
    encoded_move = np.multiply(encoded_move, encoded_legal_moves)

    # If all encoded move elements are 0, return None indicating no legal move found
    if np.all(encoded_move == 0):
        return None

    # Get the index of the max value in the encoded move and decode it
    from_square, to_square = divmod(np.argmax(encoded_move), 64)
    return chess.Move(from_square, to_square)



def train(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        #we iterate over each board with the corrrepsond move made at that board state
        for boards, moves in dataloader:
            optimizer.zero_grad()
            batch_boards = []
            batch_targets = []
            for board_fen, move in zip(boards, moves):
                board = chess.Board(board_fen)
                #convert to tensor then add it to the batch
                #unsqueeze adds another dim for the batch to be concatenated along later
                board_tensor = model.board_to_tensor(board).unsqueeze(0).to(device)
                batch_boards.append(board_tensor)
                target_vector = torch.tensor(encode_move(move), dtype=torch.float32).unsqueeze(0).to(device)
                batch_targets.append(target_vector)
            #concatenate them to form a nx12x8x8 tensor where n is the number of states the game had
            batch_boards = torch.cat(batch_boards)
            batch_targets = torch.cat(batch_targets)
            #now this can be forwarded to the network
            output = model(batch_boards)
            loss = criterion(output, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


path = __file__.replace("AI.py","data.pgn")

model = ChessModel()
#binary cross entropy loss
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
games_dataset = ChessDataset(path)


train( model,games_dataset, criterion, optimizer, num_epochs=50, device=device)
model._save_to_state_dict("model.pth")
test_game = games_dataset.games[0]

board = chess.Board()
for move in test_game.mainline_moves():
    board_tensor = model.board_to_tensor(board).unsqueeze(0).to(device)
    output:torch.Tensor = model(board_tensor)
    decoded_move = decode_move(output.cpu().detach().numpy()[0], board)

    if decoded_move is None:
        print("Illegal move detected. Skipping.")
        continue
    
    # Check if the decoded move is legal on the current board
    if decoded_move in board.legal_moves:
        print(f"Predicted move: {decoded_move}, Actual move: {move}")
        board.push(decoded_move)
    else:
        print(f"Invalid move predicted: {decoded_move}. Skipping.")
