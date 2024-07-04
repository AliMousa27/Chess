#IMPORTANT: The AI is not good enough. I have trained it and validated it against stockfish and it played at about a player with elo of 1000. Therefore i will move to implement a
#traditional minimax algorithm with alpha beta pruning. If you can spot the mistake thats making the model converge way earlier than it should 
#(it should ideally converge with absoulte loss of at most 20 but its converging at 50) kindly let me know.


import chess
import torch
import torch.nn as nn
from torch import device, optim
import time
from data_handler import ChessDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from stockfish import Stockfish

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += identity
        return self.relu(out)


class ChessModel(nn.Module):
    def __init__(self, num_blocks=70, channels=128):
        super(ChessModel, self).__init__()
        self.initial_conv = ConvBlock(12, channels, kernel_size=3, stride=1, padding=1)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_tower(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        board_matrix = np.zeros((8, 8, 12), dtype=np.float32)
        piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        piece_values = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 10,
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                plane = piece_to_plane[piece.symbol()]
                row, col = divmod(square, 8)
                board_matrix[row, col, plane] = piece_values.get(piece.symbol().lower(), 0)
                # Min-Max normalization
        board_tensor = torch.tensor(board_matrix, dtype=torch.float32)
        min_val = board_tensor.min()
        max_val = board_tensor.max()
        board_tensor = (board_tensor - min_val) / (max_val - min_val)
        return board_tensor.permute(2, 0, 1)



def train(model, criterion, optimizer, num_epochs, device, dataloader):
    model.to(device)
    model.train()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(dataloader), epochs=num_epochs)
    total_entries = len(dataloader.dataset)
    print(f"Using device: {device}")
    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        processed_entries = 0
        for i, (board_fens, eval) in enumerate(dataloader):
            optimizer.zero_grad()
            board_tensors = torch.stack([model.board_to_tensor(chess.Board(fen)) for fen in board_fens]).to(device)
            output = model(board_tensors)
            eval = eval.float().view(-1, 1).to(device)
            loss = criterion(output, eval)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            processed_entries += len(board_fens)
   
            if (i + 1) % 10 == 0:
                print(f"I IS {i}")
                print(f"Epoch {epoch}/{num_epochs}, average loss: {total_loss / (i + 1):.4f}.")
                print(f"Processed {processed_entries}/{total_entries} Time taken {time.time() - start_time} seconds.\n")
                start_time = time.time()
        print(f"Epoch {epoch}/{num_epochs}, average loss: {total_loss / i+1:.4f}. Time taken {time.time() - start_time} seconds.")
        start_time = time.time()

data_loader = DataLoader(ChessDataset(r"C:\Users\Jafar\Desktop\Chess\evals.json"), batch_size=1, shuffle=True)
model = ChessModel()
#model.load_state_dict(torch.load("model.pth"))
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, criterion, optimizer, 200, device, data_loader)
print("Training completed")
torch.save(model.state_dict(), "model.pth")



def choose_best_move(model: ChessModel, board: chess.Board,device) -> chess.Move:
    best_move = None
    best_eval = float('-inf')
    for move in board.legal_moves:
        board.push(move)
        board_tensor = model.board_to_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(board_tensor).item()
        if best_move is None or output > best_eval:
            best_move = move
            best_eval = output
        board.pop()
    return best_move

def validate(model,stockfish):
    model.eval()
    board = chess.Board()
    stockfish.set_fen_position(board.fen())
    corrects=0
    for i in range(100):
        board_tensor = model.board_to_tensor(board).unsqueeze(0)
        predicted_eval = model(board_tensor).item()
        actual_eval = stockfish.get_evaluation()
        print(f"Predicted: {predicted_eval}, Actual: {actual_eval}")
        if actual_eval > 0:
            lower_bound = actual_eval * 0.8
            upper_bound = actual_eval * 1.2
        else:
            lower_bound = actual_eval * 1.2
            upper_bound = actual_eval * 0.8
        if lower_bound <= predicted_eval <= upper_bound:
            corrects+=1
        print(f"Corrects: {corrects}. Accuracy {corrects/(i+1)}")