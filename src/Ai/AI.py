import json
import threading
import chess
import numpy as np
import torch
import torch.nn as nn
import chess.pgn
from torch import optim
import time
from data_handler import ChessDataset


data_path = __file__.replace("AI.py","better_data.jsonl")
model_to_save_path = __file__.replace("AI.py","checkpoint.pth")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(40)])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 64 * 64)
        )

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
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

def get_bestmove_and_board( game):

        board = chess.Board(game["fen"])
        evals = game["evals"]
        highest_cp = -float("inf")
        best_line = None
        for eval in evals:
            for pvs in eval["pvs"]:
                if "cp" in pvs and pvs["cp"] > highest_cp:
                    highest_cp = pvs["cp"]
                    best_line = pvs["line"]

        if best_line is None:
            return None, None
        from_square = chess.parse_square(best_line[0:2])
        to_square = chess.parse_square(best_line[2:4])

        return board, chess.Move(from_square, to_square)
def train(model, criterion, optimizer, num_epochs, device):
    model.to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print(f"Using device: {device}")
    
    stop_training_flag = threading.Event()
    input_thread = threading.Thread(target=monitor_input, args=(stop_training_flag,))
    input_thread.start()
    
    try:
        model, optimizer, start_epochs, line_index, loss = load_checkpoint(model, optimizer, model_to_save_path)
        print(f"Model loaded from {model_to_save_path}. Starting from epoch {start_epochs}, line {line_index}, loss {loss}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch")
        line_index = 0
        start_epochs = 0
        loss = 0.0
    
    avg_loss = 0.0
    for epoch in range(start_epochs,num_epochs):
        if stop_training_flag.is_set():
            print("Quitting training...")
            break
        start_time = time.time()
        total_loss = 0.0
        
        with open(r"C:\Users\Jafar\Downloads\better_data.jsonl", 'r') as file:
            for i, line in enumerate(file):
                if stop_training_flag.is_set():
                    print("Quitting training...")
                    save_checkpoint(model, optimizer, model_to_save_path, epoch, i, loss)
                    break
                if i < line_index:
                    continue
                board, best_move = get_bestmove_and_board(json.loads(line))
                if board is None or best_move is None:
                    continue
                optimizer.zero_grad()
                board_tensor = model.board_to_tensor(board).unsqueeze(0).to(device)
                target_vector = torch.tensor(encode_move(best_move), dtype=torch.float32).unsqueeze(0).to(device)
                output = model(board_tensor)
                loss = criterion(output, target_vector)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 1000 == 0:
                    avg_loss = total_loss / (i + 1)
                    print(f"Epoch {epoch}/{num_epochs}, I: {i}, Loss: {avg_loss:.4f}")
                    
            scheduler.step()
            avg_loss = total_loss / (i + 1)
            end_time = time.time()
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}. Time taken: {end_time-start_time:.2f}s")
            
            


                
def save_checkpoint(model, optimizer, path, epoch,line_index,loss):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'line_index':line_index,
        'loss':loss
    }, path)
    print(f"Checkpoint saved to {path}. Epoch: {epoch}, Line: {line_index}, Loss: {loss}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    line_index = checkpoint['line_index']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path}. Epoch: {epoch}, Line: {line_index}, Loss: {loss}")
    return model, optimizer, epoch, line_index, loss


def monitor_input(stop_training_flag):
    while True:
        if input() == 'quit':
            stop_training_flag.set()
            break

model = ChessModel()

#model=torch.load(r"c:\Users\k876y\OneDrive\Desktop\chess\src\Ai\model.pth")


#binary cross entropy loss
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#start_time = time.time()
#games_dataset = ChessDataset(r"C:\Users\Jafar\Downloads\better_data.jsonl")
#end_time=time.time()
#print(f"Time taken to load the dataset is {end_time-start_time}")

train( model, criterion, optimizer,10, device)

print("end of training")