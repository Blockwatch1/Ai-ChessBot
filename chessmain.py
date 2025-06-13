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
#functions section
def makeBoardImage(board):
    with open("chess_board.svg", "w") as f:
        f.write(chess.svg.board(board=board))
# 
def cleanFenForInput(fen):
    parts = fen.split(' ')
    positions=parts[0].split('/')
    side=parts[1]
    positionsVector= []
    for i in range(len(positions)):
        row=decodeSingleFenRow(positions[i],side)
        positionsVector.extend(row)
    positionsVector=np.array(positionsVector)
    return positionsVector
def get_sign(number):
  if number > 0:
    return 1
  else: 
    return -1
def valueOf(piece, side):
    if side == 'w':
        if piece == 'p':
            return 1
        elif piece == 'n':
            return 2
        elif piece == 'b':
            return 3
        elif piece == 'r':
            return 4
        elif piece == 'q':
            return 5
        elif piece == 'k':
            return 9
        elif piece == 'P':
            return 1 + 9
        elif piece == 'N':
            return 2 + 9
        elif piece == 'B':
            return 3 + 9
        elif piece == 'R':
            return 4 + 9
        elif piece == 'Q':
            return 5 + 9
        elif piece == 'K':
            return 9 + 9
    elif side == 'b':
        if piece == 'P':
            return 1
        elif piece == 'N':
            return 2
        elif piece == 'B':
            return 3
        elif piece == 'R':
            return 4
        elif piece == 'Q':
            return 5
        elif piece == 'K':
            return 9
        elif piece == 'p':
            return 1 + 9
        elif piece == 'n':
            return 2 + 9
        elif piece == 'b':
            return 3 + 9
        elif piece == 'r':
            return 4 + 9
        elif piece == 'q':
            return 5 + 9
        elif piece == 'k':
            return 9 + 9
     
def canCastle(row,castleRights,side):
    king='k'
    queen='q'
    if side=='w':
        king='K'
        queen='Q'
    canCastleKingSide=1;
    if king not in castleRights:
        canCastleKingSide=0;
    subrow = row[5:7]
    if any(x != 0 for x in subrow):
        canCastleKingSide = 0
    canCastleQueenSide= 1
    if queen not in castleRights:
        canCastleQueenSide=0
    subrow = row[1:4]
    if any(x != 0 for x in subrow):
        canCastleQueenSide=0
    return (canCastleKingSide,canCastleQueenSide)

    
def decodeSingleFenRow(row,side):
    decodedRow=[]
    if row=='8':
        return [0]*8
    cleanedRow= cleanFenRow(row,side)
    i=0
    while i< len(cleanedRow):
        if cleanedRow[i]==0:
            decodedRow.append(0)
            i+=1
        else:
            decodedRow.append(valueOf(cleanedRow[i],side))
            i+=1
    return decodedRow
def cleanFenRow(row,side):
    cleaned =[]
    for i in range(0,len(row)):
        if is_number(row[i]):
            cleaned.extend([0]*int(row[i]))
        else:
            cleaned.append(row[i])
    return cleaned
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
   
def checkForPromotionNeed(state,action,model,side):
    from_sq= chess.square_name(action.from_square)  
    to_sq=chess.square_name(action.to_square)
    from_ind= model.local_network.squares.index(from_sq)   
    to_ind= model.local_network.squares.index(to_sq)
    temp=state[from_ind]
    tempState=list(state)
    tempState[to_ind]=temp
    promotion=False
    if side==0:
        for i in range(0,8):
            if tempState[i]==10:
                return True
    if side==1:
        for i in range(64-8,64):
            if tempState[i]==10:
                return True
    return promotion
    
    
    
#functions section ends

stockfish = Stockfish(path="stockfish\\stockfish-windows-x86-64-avx2.exe")
opponent = Stockfish(path="stockfish\\stockfish-windows-x86-64-avx2.exe")
opponent.set_elo_rating(500)
opponent.update_engine_parameters({
    "UCI_LimitStrength": True,
    "UCI_Elo": 500
})

# --- Corrected section to check the ELO ---
print("\n--- Checking Stockfish Opponent ELO ---")
try:
    # Get all UCI options
    options = opponent.get_parameters()
    
    # Check if 'UCI_LimitStrength' is True and 'UCI_Elo' is set correctly
    limit_strength_set = False
    elo_set = False
    
    # Retrieve the values, defaulting to None if not present
    actual_limit_strength = options.get("UCI_LimitStrength")
    actual_elo = options.get("UCI_Elo")

    # Check UCI_LimitStrength
    if isinstance(actual_limit_strength, str):
        if actual_limit_strength.lower() == "true":
            limit_strength_set = True
    elif isinstance(actual_limit_strength, bool):
        if actual_limit_strength is True:
            limit_strength_set = True
    
    print(f"UCI_LimitStrength: {actual_limit_strength}")

    # Check UCI_Elo
    if isinstance(actual_elo, str):
        if actual_elo == "500":
            elo_set = True
    elif isinstance(actual_elo, (int, float)): # In case it gets parsed as a number
        if actual_elo == 500:
            elo_set = True
    
    print(f"UCI_Elo: {actual_elo}")
    
    if limit_strength_set and elo_set:
        print("SUCCESS: Stockfish opponent should be limited to 500 ELO.")
    else:
        print("WARNING: Stockfish ELO settings might not be applied correctly (internal check failure).")
        # Print the actual values obtained from get_parameters() to see what they are
        print(f"  Actual UCI_LimitStrength value: {actual_limit_strength} (Type: {type(actual_limit_strength)})")
        print(f"  Actual UCI_Elo value: {actual_elo} (Type: {type(actual_elo)})")

except Exception as e:
    print(f"Error checking Stockfish parameters: {e}")
    print("Ensure the Stockfish path is correct and the engine is executable.")

print("--------------------------------------\n")

currSide=1 #0 for white , 1 for black
env = gym.make("Chess-v0")
total_moves_to_make=30000
anti_epsilon= 1
epsilon_decay=0.999
done=0
learning_rate=0.002
capacity=400
randomising=True
mate_base_reward=40
score_over_40_moves=collections.deque(maxlen=40)
obs = env.reset()
model= Agent(capacity,learning_rate)
try :
    for i in range(total_moves_to_make):
        env.render()
        if currSide==0:
            legal_moves = list(env.legal_moves)
            state = cleanFenForInput(obs.fen())
            stockfish.set_fen_position(obs.fen())
            before_evaluation=stockfish.get_evaluation()
            before_evaluation= before_evaluation['value']/100 if before_evaluation['type']=='cp' else get_sign(before_evaluation['value'])*mate_base_reward-before_evaluation['value']
            action=model.act(state,legal_moves,anti_epsilon,randomizing=randomising)
            needPromotion= checkForPromotionNeed(state,action,model,currSide)
            promotionTensor=None
            if needPromotion:
                from_sq= chess.square_name(action.from_square)  
                to_sq=chess.square_name(action.to_square)
                promotionTensor = model.promotion_network(state)
                promotion='' 
                ind= torch.argmax(promotionTensor)
                if ind==0:
                    promotion='q'
                elif ind==1:
                    promotion='n'
                elif ind==2:
                    promotion='b'
                else:
                    promotion='r'   
                print(from_sq+to_sq+promotion)
                print(legal_moves)
                action=chess.Move.from_uci(from_sq+to_sq+promotion)    
            obs, _, done, info = env.step(action)
            if done:
                result = obs.result()
                print(result)
                winner=None
                if result == "1-0":
                    winner = 0  # white
                elif result == "0-1":
                    winner = 1  # black
                else:
                    winner = None 
                reward=0
                print("side","white" if currSide==0 else "black")
                if winner==currSide:
                     reward=mate_base_reward+10
                elif winner==1:
                    reward=-mate_base_reward-10
                else:
                    reward=0
                if needPromotion:
                    model.teachPromotionModel(promotionTensor,reward)  
                model.step(state,from_ind,to_ind,legal_moves,reward)
                currSide=1
                score_over_40_moves.append(reward)
                anti_epsilon=anti_epsilon*epsilon_decay
                obs=env.reset()
                continue
                
            
                
            fen = obs.fen()
            opponent.set_fen_position(obs.fen())
            opponent_move = chess.Move.from_uci(opponent.get_best_move())
            obs, _, done, info = env.step(opponent_move)
            if done:
                print(info,"info")
                result = obs.result()
                print(result)
                winner=None
                if result == "1-0":
                    winner = 0  # white
                elif result == "0-1":
                    winner = 1  # black
                else:
                    winner = None  # draw
                reward=0
                print("side","white" if currSide==0 else "black")
                if winner==currSide:
                    reward=mate_base_reward+10
                elif winner==1:
                    reward=-mate_base_reward-10
                else:
                    reward=0
                if needPromotion:
                    model.teachPromotionModel(promotionTensor,reward)  
                model.step(state,from_ind,to_ind,legal_moves,reward)
                currSide=1
                score_over_40_moves.append(reward)
                anti_epsilon=anti_epsilon*epsilon_decay
                obs=env.reset()
                continue
            fen=obs.fen()
            stockfish.set_fen_position(fen)
            after_evaluation=stockfish.get_evaluation()
            after_evaluation= after_evaluation['value']/100 if after_evaluation['type']=='cp' else get_sign(after_evaluation['value'])*mate_base_reward-after_evaluation['value']
            reward=after_evaluation-before_evaluation
            score_over_40_moves.append(reward)
            from_sq= chess.square_name(action.from_square)  
            to_sq=chess.square_name(action.to_square) 
            from_ind= model.local_network.squares.index(from_sq)   
            to_ind= model.local_network.squares.index(to_sq) 
            if needPromotion:
                model.teachPromotionModel(promotionTensor,reward)  
            model.step(state,from_ind,to_ind,legal_moves,reward)
            anti_epsilon=anti_epsilon*epsilon_decay
        if currSide==1:
            # legal_moves = list(env.legal_moves)
            # state = cleanFenForInput(obs.fen())
            stockfish.set_fen_position(obs.fen())
            before_evaluation=stockfish.get_evaluation()
            before_evaluation= -before_evaluation['value']/100 if before_evaluation['type']=='cp' else -get_sign(before_evaluation['value'])*mate_base_reward+before_evaluation['value']
            opponent.set_fen_position(obs.fen())
            opponent_move = chess.Move.from_uci(opponent.get_best_move())
            obs, _, done, info = env.step(opponent_move)
            # action=model.act(state,legal_moves,anti_epsilon,randomizing=randomising)
            # needPromotion= checkForPromotionNeed(state,action,model,currSide)
            # promotionTensor=None
            # if needPromotion:
            #     from_sq= chess.square_name(action.from_square)  
            #     to_sq=chess.square_name(action.to_square)
            #     promotionTensor = model.promotion_network(state)
            #     promotion='' 
            #     ind= torch.argmax(promotionTensor)
            #     if ind==0:
            #         promotion='q'
            #     elif ind==1:
            #         promotion='n'
            #     elif ind==2:
            #         promotion='b'
            #     else:
            #         promotion='r'   
            #     move=chess.Move.from_uci(from_sq+to_sq+promotion)    
            # obs, _, done, info = env.step(action)
            if done:
                result = obs.result()
                print(result)
                winner=None
                if result == "1-0":
                    winner = 0  # white
                elif result == "0-1":
                    winner = 1  # black
                else:
                    winner = None 
                reward=0
                if winner==currSide:
                     reward=mate_base_reward+10
                elif winner==0:
                     reward=-mate_base_reward-10
                else:
                    reward=0
                print("side","white" if currSide==0 else "black")
                if needPromotion:
                    model.teachPromotionModel(promotionTensor,reward)  
                model.step(state,from_ind,to_ind,legal_moves,reward)
                currSide=0
                score_over_40_moves.append(reward)
                anti_epsilon=anti_epsilon*epsilon_decay
                obs=env.reset()
                continue
                
            fen = obs.fen()
            legal_moves = list(env.legal_moves)
            state = cleanFenForInput(obs.fen())
            action=model.act(state,legal_moves,anti_epsilon,randomizing=randomising)
            needPromotion= checkForPromotionNeed(state,action,model,currSide)
            promotionTensor=None
            if needPromotion:
                from_sq= chess.square_name(action.from_square)  
                to_sq=chess.square_name(action.to_square)
                promotionTensor = model.promotion_network(state)
                promotion='' 
                ind= torch.argmax(promotionTensor)
                if ind==0:
                    promotion='q'
                elif ind==1:
                    promotion='n'
                elif ind==2:
                    promotion='b'
                else:
                    promotion='r'   
                action=chess.Move.from_uci(from_sq+to_sq+promotion)    
            obs, _, done, info = env.step(action)
            if done:
                result = obs.result()
                print(result)
                winner=None
                if result == "1-0":
                    winner = 0  # white
                elif result == "0-1":
                    winner = 1  # black
                else:
                    winner = None  # draw
                print("side","white" if currSide==0 else "black")
                reward=0
                if winner==currSide:
                    reward= mate_base_reward+10
                elif winner==0:
                     reward=-mate_base_reward-10
                else:
                    reward=0
                if needPromotion:
                    model.teachPromotionModel(promotionTensor,reward)  
                model.step(state,from_ind,to_ind,legal_moves,reward)
                currSide=0
                score_over_40_moves.append(reward)
                anti_epsilon=anti_epsilon*epsilon_decay
                obs=env.reset()
                continue
            fen=obs.fen()
            stockfish.set_fen_position(fen)
            after_evaluation=stockfish.get_evaluation()
            after_evaluation= -after_evaluation['value']/100 if after_evaluation['type']=='cp' else -get_sign(after_evaluation['value'])*mate_base_reward+after_evaluation['value']
            reward=after_evaluation-before_evaluation
            score_over_40_moves.append(reward)
            from_sq= chess.square_name(action.from_square)  
            to_sq=chess.square_name(action.to_square) 
            from_ind= model.local_network.squares.index(from_sq)   
            to_ind= model.local_network.squares.index(to_sq) 
            if needPromotion:
                model.teachPromotionModel(promotionTensor,reward)  
            model.step(state,from_ind,to_ind,legal_moves,reward)
            anti_epsilon=anti_epsilon*epsilon_decay
            makeBoardImage(obs)
            

        print("Score",np.mean(score_over_40_moves))
            
    env.close()
except Exception as e:
    print(e)
path="agent_checkpoint.pth"
torch.save({
    'local_network_state_dict': model.local_network.state_dict(),
    'promotion_network_state_dict':model.promotion_network.state_dict(),
    'optimiser_state_dict': model.optimiser.state_dict(),
    'promotion_optimiser_state_dict': model.promotion_network_optimiser.state_dict(),
    'memory': model.memory,  # Only if ReplayMemory is serializable
    't_step': model.t_step
}, path)


