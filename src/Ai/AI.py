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

