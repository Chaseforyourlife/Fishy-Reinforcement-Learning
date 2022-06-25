import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet,QTrainer

BATCH_SIZE = 1_000
LEARNING_RATE = 0.0001
MAX_MEMORY = 100_000
STARTING_EPSILON = 400
GAMMA = 0.9 # must be less than 1
INPUT_SIZE = 46
HIDDEN_SIZE = 256
OUTPUT_SIZE = 5



def calculate_reward(fishy,fish_eaten,win):
    reward = 0
    if fishy.alive:
        reward += 1
    else:
        reward -= 50
    reward += fish_eaten*50
    if win:
        reward += 100000
    return reward



class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0   # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() when exceeding max_memory
        #TODO: model,trainer
        self.model = Linear_QNet(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE)
        self.trainer = QTrainer(self.model,LEARNING_RATE,GAMMA)
    def get_state(self,fishy,school):
        game_state = []
        #Input Layer Data
        #TODO ADD DISTANCE FROM SIDES OF SCREEN ???
        game_state.append(fishy.x)
        game_state.append(fishy.y)
        game_state.append(fishy.width)
        game_state.append(fishy.height)
        game_state.append(fishy.x_speed)
        game_state.append(fishy.y_speed)
        #Add data for 8 fish
        for fish in school.fish_list:
            game_state.append(fish.x)
            game_state.append(fish.y)
            game_state.append(fish.width)
            game_state.append(fish.height)
            game_state.append(fish.x_speed)
        return np.array(game_state,dtype=float)

    def remember(self,state,acion,reward,next_state,done):
        self.memory.append((state,acion,reward,next_state,done))
    
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)
    
    
    def get_action(self,state):
        #random moves: tradeoff exporation/exploitation
        self.epsilon = STARTING_EPSILON - self.n_games
        #get empty list of 0s to replace with move
        move = [0,0,0,0,0]
        #Could change what goes into the randint max
        if random.randint(0,STARTING_EPSILON) < self.epsilon:
            move_index = random.randint(0,4)
            move[move_index] = 1
        else:
            state_tensor = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state_tensor)   # Calls Forward Function of LinearQModel
            move_index = torch.argmax(prediction).item()
            move[move_index] = 1
        return move




