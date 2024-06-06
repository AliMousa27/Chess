from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from .piece_color import Piece_Color
class Piece(metaclass=ABCMeta):
  """
  constructor to create a given piece
  Args:
    name: the name of the piece to be instantiated
    position: the i ital position consisting of a tuple of 2 ints, row then column
    color: Enum to indicate the color
  """
  @abstractmethod
  def __init__(self,name:str,position:tuple,color: Piece_Color) -> None:
    super().__init__()
    self.name=name
    self.position=position
    self.color = color
    
    
  """
  method to calculate all possible moves regardless if they are valid or not
  
  Return:
	a list of tuples where the piece can move in terms of rows and columns
  """
  @abstractmethod
  def calc_all_moves(self) -> List[Tuple]:
    pass

  """
  method to change the positon of the piece
  returns nothing
  """
  @abstractmethod
  def move(self)-> None: 
    pass

