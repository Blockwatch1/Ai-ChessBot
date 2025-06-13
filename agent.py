
import numpy as np
import torch
from chessnetwork import Network,PromotionNetwork
import torch.optim as optim
import random
class Agent():

  def __init__(self, memory_size,learning_rate):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.local_network = Network().to(self.device)
    self.promotion_network = PromotionNetwork().to(self.device)
    self.optimiser = optim.Adam(self.local_network.parameters(), lr = learning_rate)
    self.promotion_network_optimiser= optim.Adam(self.promotion_network.parameters(),lr=learning_rate)
    self.memory = ReplayMemory(memory_size)
    self.t_step = 0

  def step(self,state,from_index,to_index,legal_moves,reward):
    self.memory.addToMemory((state,from_index,to_index,reward,legal_moves))
    self.t_step=(self.t_step+1)%4
    if self.t_step==0 :
      sample = self.memory.sample(len(self.memory.memory))
      self.learn(sample)
      
  
  def act(self,state,legal_moves,epsilon=0.,randomizing=True):
    state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
    self.local_network.eval()
    with torch.no_grad():
      action,_,_,_,_= self.local_network(state,legal_moves)
      print(action,"action")
    self.local_network.train()
    number = np.random.randn()
    if randomizing:
        if number < epsilon:
            action = np.random.choice(legal_moves)
    return action


  # def learn (self,sample):
  #   states,past_from,past_to,legal_moves,rewards= sample
  #   actions,from_indices,to_indices,from_probs,to_probs= self.local_network(states,legal_moves)
  #   loss= -(torch.log(from_probs[past_from])+torch.log(to_probs[past_to]))*rewards 
  #   self.optimiser.zero_grad()
  #   loss.backward()
  #   self.optimiser.step()
  def learn(self, sample):
    states, past_from, past_to, rewards, legal_moves = sample
    from_probs=[]
    to_probs=[]
    for state,legal_moves_in_turn in  zip(states, legal_moves):
      _, from_indices, to_index, from_prob, to_prob = self.local_network(state, legal_moves_in_turn)
      from_probs.append(from_prob)
      to_probs.append(to_prob)
      
    from_probs = torch.stack(from_probs)
    to_probs= torch.stack(to_probs)
    past_from = past_from.long()
    past_to = past_to.long()

    batch_size = rewards.size(0)

    selected_from_logprobs = torch.log(from_probs[torch.arange(batch_size), past_from.squeeze()])
    selected_to_logprobs = torch.log(to_probs[torch.arange(batch_size), past_to.squeeze()])

    loss = (-(selected_from_logprobs + selected_to_logprobs) * rewards.squeeze()).mean()

    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()

  def teachPromotionModel(self,tensor,reward):
    loss= -torch.log(torch.max(tensor))*reward
    loss=loss.mean()
    self.promotion_network_optimiser.zero_grad()
    loss.backward()
    self.promotion_network_optimiser.step()
    
class ReplayMemory():
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def addToMemory(self,memory):
        self.memory.append(memory)
        if len(self.memory)>=self.capacity:
            del self.memory[0]
    # def sample(self,sample_size):
    #     memory= self.memory
    #     np.random.shuffle(memory)
    #     sample= memory[0:sample_size]
    #     states,past_from_indices,past_to_indices,rewards,past_legal_moves = [],[],[],[],[]
    #     for memory in sample:
    #         if memory is None:
    #             continue
    #         states.append(memory[0])
    #         past_from_indices.append(memory[1])
    #         past_to_indices.append(memory[2])
    #         rewards.append(memory[3])
    #         past_legal_moves.append(memory[4])
    #     for s in states:
    #       print(type(s), getattr(s, 'shape', 'no shape'))

    #     states=torch.from_numpy(np.vstack(states)).float().to(self.device)
    #     past_from_indices=torch.from_numpy(np.vstack(past_from_indices)).float().to(self.device)
    #     past_to_indices=torch.from_numpy(np.vstack(past_to_indices)).float().to(self.device)
    #     rewards= torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    #     past_legal_moves= np.vstack(past_legal_moves)
    #     return states,past_from_indices,past_to_indices,rewards
    def sample(self, sample_size):
        # Sample without modifying original memory order
      sample = random.sample(self.memory, sample_size)


          # Initialize lists
      states, past_from_indices, past_to_indices, rewards, past_legal_moves = [], [], [], [], []

      for m in sample:
          if m is None:
              continue
          states.append(m[0])
          past_from_indices.append(m[1])
          past_to_indices.append(m[2])
          rewards.append(m[3])
          past_legal_moves.append(m[4])

          # Stack and convert to tensors
      states = np.vstack(states)
      past_from_indices = torch.from_numpy(np.vstack(past_from_indices)).float().to(self.device)
      past_to_indices = torch.from_numpy(np.vstack(past_to_indices)).float().to(self.device)
      rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device) 

      return states, past_from_indices, past_to_indices, rewards, past_legal_moves  
          
      