from abc import ABCMeta, abstractmethod
import random
from typing import List, Tuple
from .piece_color import Piece_Color
import pygame

class Piece(metaclass=ABCMeta):

  @abstractmethod
  def __init__(self,name:str,position:tuple,color: Piece_Color,img_size:int) -> None:
    """
    constructor to create a given piece
    Args:
      name: the name of the piece to be instantiated
      position: the inital position consisting of a tuple of 2 ints, row then column
      color: Enum to indicate the color
    """
    super().__init__()
    self.name=name
    self.position=position
    self.color = color
    color_str = "white" if color ==Piece_Color.WHITE else "black"
    image = pygame.image.load(fr'src/assets/{name.lower()}_{color_str}.png')
    #scale down the image and assign the key to that value
    self.image= pygame.transform.scale(image, (img_size, img_size)) 
    self.possible_moves = []
    self.id = random.randint(0,1000000)

    
    

  @abstractmethod
  def calc_all_moves(self) -> List[Tuple]:
    """
    method to calculate all possible moves regardless if they are valid or not
    
    Return:
    a list of tuples where the piece can move in terms of rows and columns
    """
    pass
  
  @abstractmethod
  def filter_moves(self,board: List[List],is_pinned,check_for_pins:bool) -> List[Tuple]:
    pass
