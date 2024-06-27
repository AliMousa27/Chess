import chess
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, path:str):
        print(f"The path is {path}")
        self.games = self.load_games(path)
    
    def __len__(self):
        return len(self.games)
    #the get item will return a list of the board states along with the move made at each state
    def __getitem__(self, idx):
        game = self.games[idx]
        board = chess.Board()
        states, moves = [], []
        for move in game.mainline_moves():
            states.append(board.fen())
            moves.append(move)
            board.push(move)
        return states, moves
    
    def collate_batch(batch):
        boards, moves = [], []
        for states, moves_ in batch:
            boards.extend(states)
            moves.extend(moves_)
        return boards, moves
      
    def load_games(self,path,limit=10):
        games = []
        with open(path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        return games