# ... (your existing imports and functions) ...
import gym
import gym_chess
import numpy as np
import chess  # comes with python-chess
import chess.svg
from IPython.display import SVG, display
import random
from stockfish import Stockfish
from chessnetwork import Network
from agent import Agent
import collections
import torch
# Initialize Stockfish opponent as you currently do

# Set the ELO
# ... (your existing imports and functions) ...

# Initialize Stockfish opponent as you currently do
# ... (your existing imports and functions) ...

# Initialize Stockfish opponent as you currently do
opponent = Stockfish(path="stockfish\\stockfish-windows-x86-64-avx2.exe")

# Set the ELO (this part is correct)
fen ="8/1KPppppP/2Pppppp/2Ppppp1/2PPPPPP/8/2QQQQ2/7k w - - 0 1"
opponent.set_fen_position(fen)
eval=opponent.get_evaluation()
print(eval)
# ... (rest of your code) ...

# ... (rest of your code) ...

# ... (rest of your code) ...