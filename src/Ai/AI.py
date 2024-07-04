<<<<<<< HEAD
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
=======
from calendar import c
import enum
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
from torch.utils.data import DataLoader


data_path = __file__.replace("AI.py","better_data.jsonl")
model_to_save_path = __file__.replace("AI.py","checkpoint.pth")


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(2048,1),
        )

    def forward(self, x):
        return self.model(x)

        

    
    def board_to_tensor(self,board: chess.Board) -> torch.Tensor:
        # create 8x8x12 tensor (8x8 board, 12 planes(type of piece and color is the plane))
        #so 12 lists for each type of piece each list is a 2d 8x8 matrix
        board_matrix = np.zeros((8, 8, 12), dtype=np.int8)
        #create a dict to map the piece to the plane. This 
>>>>>>> ecc242158a352f661b7b9fec7a9a029434ac0243
        piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
<<<<<<< HEAD
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
=======
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





def train(model, criterion, optimizer, num_epochs, device,dataloader):
    model.to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print(f"Using device: {device}")
    
    i =0
    start_time = time.time()
    for epoch in range(num_epochs):

        total_loss = 0.0
        
        for board_fen, eval in dataloader:
                '''if stop_training_flag.is_set():
                    print("Quitting training...")
                    save_checkpoint(model, optimizer, model_to_save_path, epoch, 0, loss)
                    break'''
                # TODO if board is None or best_move is None:
                    #continue
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
                i+=1
                if i % 10000 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, average loss: {total_loss / i:.4f} Date entry number {i} out of {len(data_loader)}.")
                    print(f"normalized output: {normalized_output}, normalized eval: {normalized_eval}")
                    print(f"output: {output}, eval: {eval}")
                    print()
        print(f"Epoch {epoch}/{num_epochs}, average loss: {total_loss / i:.4f}. Time taken {time.time()-start_time} seconds.")
        scheduler.step()
            
            


                
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


data_loader = DataLoader(ChessDataset(r"C:\Users\Jafar\Desktop\Chess\evals.json"), batch_size=1, shuffle=False)

model = ChessModel()


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train( model, criterion, optimizer,1000, device,data_loader)
torch.save(model.state_dict(), r"model.pth")
print(f"saved")

>>>>>>> ecc242158a352f661b7b9fec7a9a029434ac0243
