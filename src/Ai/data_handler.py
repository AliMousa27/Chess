from turtle import st
from typing import Dict, List, Tuple
import chess
import chess.pgn
from torch.utils.data import Dataset
import json
from stockfish import Stockfish

def pgn_to_json(path: str) -> None:
    #TODO use environment variable for stockfish path
    stockfish = Stockfish(path=r"C:\Users\Jafar\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
    with open(path,"r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_moves_data: List[Tuple[str, int]] = []
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                stockfish.set_fen_position(board.fen())
                # Assuming evaluation is 0 for simplicity; replace or compute as needed
                game_moves_data.append((board.fen(), stockfish.get_evaluation()["value"]))
            
            with open('games_data.json', 'a') as json_file:
                json.dump(game_moves_data, json_file, indent=4)
                
class ChessDataset(Dataset):
    def __init__(self,data_path:str) -> None:
        self.data = self.load_data(data_path)
        
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.data[idx]
    
    def load_data(self,path: str) -> List[Tuple[str, int]]:
        with open(path, 'r') as json_file:
            return json.load(json_file)


            