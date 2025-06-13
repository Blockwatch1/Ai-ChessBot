import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import chess
class Network(nn.Module):

  def __init__(self, seed = 42):
    super(Network, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.input = nn.Linear(64, 128)
    self.layer1 = nn.Linear(128, 128)
    self.layer2 = nn.Linear(128,128)
    self.layer3 = nn.Linear(128,64)
    self.fromSquare=  nn.Linear(64,64)
    self.toSquare= nn.Linear(64,64)
    self.squares = [f"{file}{rank}" for rank in range(8, 0, -1) for file in "abcdefgh"]
    
    

  def forward(self, state, legal_moves):
    state = torch.tensor(state, dtype=torch.float32)
    x = self.input(state)
    x = F.relu(x)
    x = self.layer1(x)
    x= F.relu(x)
    x= self.layer2(x)
    x = F.relu(x)
    x= self.layer3(x)
    x = F.relu(x)
    fromSquare= self.fromSquare(x)
    toSquare= self.toSquare(x)
    from_mask = np.zeros(64, dtype=np.float32)
    to_mask = np.zeros(64, dtype=np.float32)
    for move in legal_moves:
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        
        from_idx = self.squares.index(from_sq)
        to_idx = self.squares.index(to_sq)
        
        from_mask[from_idx] = 1.0
        to_mask[to_idx] = 1.0
    from_mask_t = torch.tensor(from_mask, dtype=torch.bool, device=fromSquare.device)
    to_mask_t = torch.tensor(to_mask, dtype=torch.bool, device=toSquare.device)
    LARGE_NEG = -1e9
    
    fromSquare_logits_masked = fromSquare.masked_fill(~from_mask_t, LARGE_NEG)
    toSquare_logits_masked = toSquare.masked_fill(~to_mask_t, LARGE_NEG)
    
    fromSquare_probs = F.softmax(fromSquare_logits_masked, dim=-1)
    toSquare_probs = F.softmax(toSquare_logits_masked, dim=-1)
    
    from_indices = np.array(torch.argsort(fromSquare_probs, descending=True).tolist()).flatten()
    to_indices = np.array(torch.argsort(toSquare_probs, descending=True).tolist()).flatten()
    
    chosen,chosen_from_index,chosen_to_index= filterLegalMoves(self,from_indices,to_indices,legal_moves)
    return chosen,chosen_from_index,chosen_to_index,fromSquare_probs,toSquare_probs
def filterLegalMoves(self,from_indices, to_indices, legal_moves):
  chosen=None
  chosen_from_index=None
  chosen_to_index=None
  for f in from_indices:
    for t in to_indices:
      if f == t:
        continue
      move = chess.Move.from_uci(self.squares[f] + self.squares[t])
      if move in legal_moves:
        chosen=move
        chosen_from_index=f
        chosen_to_index= t
        return chosen,chosen_from_index,chosen_to_index
  return chosen,chosen_from_index,chosen_to_index
class PromotionNetwork(nn.Module):
  def __init__(self, seed = 42):
    super(PromotionNetwork, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.input = nn.Linear(64, 16)
    self.middle= nn.Linear(16,16)
    self.out= nn.Linear(16,4)
    # 4 outputs , 0 for queen , 1 for knight, 2 for bishop , 3 for rook
  def forward(self, state):
    state = torch.tensor(state, dtype=torch.float32)
    x= self.input(state)
    x=F.relu(x)
    x=self.middle(x)
    x=F.relu(x)
    x=self.out(x)
    x=F.softmax(x)
    return x