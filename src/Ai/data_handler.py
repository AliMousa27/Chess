from typing import Dict
import chess
from torch.utils.data import Dataset
import json
class ChessDataset(Dataset):
    def __init__(self, path:str):
        print(f"The path is {path}")
        self.games = self.load_games(path)
    
    def __len__(self):
        return len(self.games)
    #the get item will return a list of the board states along with the move made at each state
    def __getitem__(self, idx):
        game = self.games[idx]
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
    
    '''def collate_batch(batch):
        boards, moves = [], []
        for states, moves_ in batch:
            boards.extend(states)
            moves.extend(moves_)
        return boards, moves'''
      
    def load_games(self,file_path,limit=100000):
        games = []  
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i >=limit: 
                    break
                games.append(json.loads(line))
        return games
    