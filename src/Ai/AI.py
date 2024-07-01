
import chess
import torch
import torch.nn as nn
import chess.pgn
from torch import optim
import time
from data_handler import ChessDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F


data_path = __file__.replace("AI.py", "better_data.jsonl")
model_to_save_path = __file__.replace("AI.py", "checkpoint.pth")


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU (inplace=True),
            nn.Linear(channels // reduction, channels * 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 2 * c, 1, 1)
        Z, B = torch.split(y, c, dim=1)
        return (Z * x) + B


class ResidualBlock(nn.Module):
    def __init__(self, channels, se_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.se = SELayer(channels, reduction=se_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        out += identity
        return F.relu(out)


class ChessModel(nn.Module):
    def __init__(self, num_blocks=19, channels=256, se_channels=16):
        super(ChessModel, self).__init__()
        self.input_conv = nn.Conv2d(12, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(channels, se_channels) for _ in range(num_blocks)]
        )
        self.fc1 = nn.Linear(channels * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.input_conv(x))
        x = self.residual_tower(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        board_matrix = np.zeros((8, 8, 12), dtype=np.int8)
        piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                plane = piece_to_plane[piece.symbol()]
                row, col = divmod(square, 8)
                board_matrix[row, col, plane] = 1
        return torch.tensor(board_matrix, dtype=torch.float32).permute(2, 0, 1)


def train(model, criterion, optimizer, num_epochs, device, dataloader):
    model.to(device)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular2')

    print(f"Using device: {device}")

    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (board_fen, eval) in enumerate(dataloader):
            optimizer.zero_grad()
            board_tensor = model.board_to_tensor(chess.Board(board_fen[0])).unsqueeze(0).to(device)
            output = model(board_tensor)
            eval = eval.float().unsqueeze(0).to(device)
            normalized_output = normalize(output)
            normalized_eval = normalize(eval)
            loss = criterion(normalized_output, normalized_eval)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
            if (i + 1) % 10000 == 0:
                print(f"Epoch {epoch}/{num_epochs}, average loss: {total_loss / (i + 1):.4f}.")
                print(f"Output: {output.item()}, Eval: {eval.item()}")
                print(f"Normalized output: {normalized_output.item()}, Normalized eval: {normalized_eval.item()}")
                print(f"Time taken {time.time() - start_time} seconds.\n")
                
                start_time = time.time()
        print(f"Epoch {epoch}/{num_epochs}, average loss: {total_loss / i:.4f}. Time taken {time.time()-start_time} seconds.")
        
            
            


                
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

def normalize(x, max_val=1557, min_val=-1598, dtype=torch.float):
    normalized_0_1 = (x - min_val) / (max_val - min_val)  
    return normalized_0_1 * 2 - 1 

from stockfish import Stockfish
data_loader = DataLoader(ChessDataset(r"C:\Users\Jafar\Desktop\Chess\evals.json"), batch_size=1, shuffle=False)
stockfish = Stockfish(path=r"C:\Users\Jafar\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
model = ChessModel()
model.load_state_dict(torch.load(r"C:\Users\Jafar\Desktop\Chess\model.pth"))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train( model, criterion, optimizer,1, device,data_loader)
torch.save(model.state_dict(), r"model.pth")
print(f"saved")
exit()


def choose_best_move(model, board, choose_for_white=True):
    best_move = None
    best_eval = float('-inf') if choose_for_white else float('inf')

    # Default to the first legal move in case no move improves the evaluation
    if board.legal_moves:
        best_move = next(iter(board.legal_moves))

    for move in board.legal_moves:
        board.push(move)
        board_tensor = model.board_to_tensor(board).unsqueeze(0)
        with torch.no_grad():
            evaluation = model(board_tensor).item()
        board.pop()
        if choose_for_white and evaluation > best_eval:
            best_eval = evaluation
            best_move = move
        elif not choose_for_white and evaluation < best_eval:
            best_eval = evaluation
            best_move = move

    return best_move

def play_game(model):
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        if board.turn:  # White's turn (AI)
            print("AI is thinking...")
            move = choose_best_move(model, board)
            board.push(move)
            print(f"AI plays: {move}")
        else:  # Black's turn (Player)
            print("AI is thinking...")
            move = choose_best_move(model, board,False)
            board.push(move)
            print(f"AI plays: {move}")

    print("Game over")
    print("Result:", board.result())

# Play the game
play_game(model)

