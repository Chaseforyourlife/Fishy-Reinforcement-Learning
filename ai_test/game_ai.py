import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet,QTrainer
window_size = (550,400)
MAX_FISH = 1
MAX_FISH_SPEED = 2 #6
MIN_FISH_SPEED = 1 #2
MAX_FISH_SIZE = 0 #30 #150
MIN_FISH_SIZE = -35


TELEMETRY = False
BATCH_SIZE = 500
LEARNING_RATE = .01
MAX_MEMORY = 500#1_000_000
STARTING_MEMORY = 500#2_000
EPSILON = 1
MIN_EPSILON = 1
GAMMA = .9 #0.90 # must be less than 1

INPUT_SIZE = 2+MAX_FISH*2# 6+2+MAX_FISH*8

HIDDEN_SIZE = 64
HIDDEN2_SIZE = 32
OUTPUT_SIZE = 9

RANDOM_MOVE_INDEX = None
RANDOM_MOVES_REMAINING =  0
RANDOM_MOVES_CONSTANT = 5

FRAME_FREQUENCY = 100

def printt(*strings):
    if TELEMETRY:
        for string in strings:
            print(string)

def calculate_reward(fishy,school,fish_eaten,win,flipped,stopped):
    reward = 0
    #get reward based on distance from fish
    for fish in school.fish_list:
        
        temp_reward = 5/((abs(fishy.x-fish.x)+abs(fishy.y-fish.y)))
        if fish.fish_eaten>fishy.fish_eaten:
            reward+=-1*temp_reward
        else:
            reward+=temp_reward
    #print(flipped)

    if flipped:
        reward -=10
        pass
    if stopped:
        reward -=10
        pass
    if fishy.alive:
        reward -= 1
        pass
    else:
        reward -= 50
    reward += fish_eaten*50
    if win:
        #reward += 1000
        pass
    #print('REWARD',reward)
    return reward



class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON  # randomness
        self.min_epsilon = MIN_EPSILON
        self.gamma = GAMMA   # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() when exceeding max_memory
        #TODO: model,trainer
        self.model = Linear_QNet(INPUT_SIZE,HIDDEN_SIZE,HIDDEN2_SIZE,OUTPUT_SIZE)
        self.trainer = QTrainer(self.model,LEARNING_RATE,GAMMA)
        self.random_moves_remaining = 0
        self.random_move_index = None
    def get_state(self,fishy,school):
        #####NORMALIZE INPUTS
        game_state = []
        '''
        #Input Layer Data
        game_state.append(fishy.x/window_size[0]) #x1
        game_state.append(fishy.y/window_size[1]) #y1
        game_state.append((fishy.x + fishy.width)/window_size[0]) #x2
        game_state.append((fishy.y + fishy.height)/window_size[1]) #y2
        game_state.append(fishy.x_speed/10)
        game_state.append(fishy.y_speed/10)
        #Add data for all fish
        for fish in school.fish_list:
            game_state.append((fish.x-fishy.x)/window_size[0])
            game_state.append((fish.y-fishy.y)/window_size[1])
            game_state.append(fish.x/window_size[0]) #x1
            game_state.append(fish.y/window_size[1]) #y1
            game_state.append((fishy.x + fish.width)/window_size[0]) #x2
            game_state.append((fishy.y + fish.height)/window_size[1]) #y2
            game_state.append(fish.x_speed/10)
            game_state.append(fish.fish_eaten>fishy.fish_eaten) #is_bigger
        #Add data for Map to Fishy
        #game_state.append(fishy.x) #distance from left 
        game_state.append((window_size[0]-(fishy.x+fishy.width))/window_size[0]) #distance from right
        #game_state.append(fishy.y) #distance from up
        game_state.append((window_size[1]-(fishy.y+fishy.height))/window_size[1]) #distance from down 
       '''
       #Fishy x and y
        game_state.append((fishy.x+fishy.width/2)/window_size[0])
        game_state.append((fishy.y+fishy.height/2)/window_size[1])
        #fish x and y
        for fish in school.fish_list:
            game_state.append((fish.x+fish.width/2)/window_size[0])
            game_state.append((fish.y+fish.height/2)/window_size[1])

        return np.array(game_state,dtype=float)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def train_long_memory(self):
        printt('Train_long_memory')
        if self.epsilon > self.min_epsilon:
            self.epsilon -=.01
        '''
        if self.epsilon < 1:
            self.epsilon = 0
        else:
            self.epsilon -= .01
        '''
        '''
        if self.epsilon > self.min_epsilon:
            self.epsilon -= .01
        '''
        
        if len(self.memory) > BATCH_SIZE:
            print('LARGER THAN BATCH SIZE')
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        #TRY THIS
        #printt('MEMORY:',self.memory)
        #mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)
    
    
    def get_action(self,state):
        #random moves: tradeoff exporation/exploitation
        #self.epsilon = STARTING_EPSILON - self.n_games
        #get empty list of 0s to replace with move
        move = [0,0,0,0,0,0,0,0,0]
        #Could change what goes into the randint max
        if self.random_moves_remaining > 0:
            printt('KEEP RANDOM')
            move[self.random_move_index] = 1
            self.random_moves_remaining -=1 
        elif random.randrange(0,100)/100 < self.epsilon:
            printt('RANDOM')
            move_index = random.randint(0,8)
            move[move_index] = 1
            self.random_move_index = move_index
            self.random_moves_remaining = RANDOM_MOVES_CONSTANT
        else:
            printt('NOT RANDOM')
            state_tensor = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state_tensor)   # Calls Forward Function of LinearQModel
            move_index = torch.argmax(prediction).item()
            printt('Move_index:',move_index)
            move[move_index] = 1
        return move




