
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)
    total_entries = len(dataloader.dataset)
    print(f"Using device: {device}")
    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        processed_entries = 0
        scheduler.step(total_loss)
        for i, (board_fens, eval) in enumerate(dataloader):
            optimizer.zero_grad()
            board_tensors = torch.stack([model.board_to_tensor(chess.Board(fen)) for fen in board_fens]).to(device)
            output = model(board_tensors)
            eval = eval.float().view(-1,1).to(device)
            loss = criterion( output,  eval)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
           
            processed_entries += len(board_fens)
   
            if (i + 1) % 10000 == 0:
                print(f"Epoch {epoch}/{num_epochs}, average loss: {total_loss / (i + 1):.4f}.")
                print(f"The ai output is {output} and the eval is {eval} and the loss is {loss}")
                print(f"Processed {processed_entries}/{total_entries} Time taken {time.time() - start_time} seconds.\n")
                start_time = time.time()
        print(f"Epoch {epoch}/{num_epochs}, average loss: {total_loss / i:.4f}. Time taken {time.time()-start_time} seconds.")
        torch.save(model.state_dict(), r"model.pth")
        print(f"saved model on epoch {epoch}")
        scheduler.step(total_loss / len(dataloader))
        
data_loader = DataLoader(ChessDataset(r"C:\Users\Jafar\Desktop\Chess\evals.json"), batch_size=10, shuffle=False)
model = ChessModel()
model.load_state_dict(torch.load(r"model.pth"))
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train( model, criterion, optimizer,10, device,data_loader)
torch.save(model.state_dict(), r"model.pth")
print(f"saved")
exit()

def validate(model,data_loader):
    model.eval()
    total=0
    corrects=0
    for i, (board_fens, eval) in enumerate(data_loader):
        if i == 10000:
            break
        total+=1
        with torch.no_grad():
            board_tensor = model.board_to_tensor(chess.Board(board_fens[0]))
            output = model(board_tensor.unsqueeze(0)).item()
            print(f"The ai output is {output} and the eval is {eval}")
        if output >= 0:
            lower_bound = output * 0.8  
            upper_bound = output * 1.2  
        else:
            lower_bound = output * 1.2  
            upper_bound = output * 0.8   
        if lower_bound <= eval <= upper_bound:
            corrects += 1
        print(f"Accuracy: {corrects / total * 100:.2f}%")
