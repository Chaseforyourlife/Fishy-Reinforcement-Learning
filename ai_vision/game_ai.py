import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet,QTrainer
from variables import *
import math
import cv2 as cv


from PIL import Image

TELEMETRY = False




if(ALLOW_DIAGONALS):
    OUTPUT_SIZE = 9
else:
    OUTPUT_SIZE = 5

RANDOM_MOVE_INDEX = None
RANDOM_MOVES_REMAINING =  0
RANDOM_MOVES_CONSTANT = 1

FRAME_FREQUENCY = 100

def printt(*strings):
    if TELEMETRY:
        for string in strings:
            print(string)

def calculate_reward(fishy,school,fish_eaten,win,flipped,stopped):
    reward = 0
    #get reward based on distance from fish
    for fish in school.fish_list:
        
        temp_reward = 25/max(1,math.sqrt((abs(fishy.x-fish.x)**2+abs(fishy.y-fish.y)**2)))
        if fish.fish_eaten>fishy.fish_eaten:
            if REWARD_PROXIMITY:
                reward+=-1*temp_reward
            pass
        else:
            if REWARD_PROXIMITY:
                reward+=temp_reward
            pass
    #print(flipped)

    if flipped:
        #reward -=10
        pass
    if stopped:
        #reward -=10
        pass
    if fishy.alive:
        #reward -= 1
        pass
    else:
        if REWARD_EAT:
            reward -= 50
    if REWARD_EAT:
        reward += fish_eaten*10
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
        self.model = Linear_QNet(OUTPUT_SIZE)
        self.trainer = QTrainer(self.model,LEARNING_RATE,GAMMA)
        self.random_moves_remaining = 0
        self.random_move_index = None
        self.model.load()
        if LOAD_OPTIMIZER:
            self.trainer.load()
    def get_state(self,pygame):
        #####NORMALIZE INPUTS
        game_state = []
        imgdata = pygame.surfarray.array3d(pygame.display.get_surface())
        #imgdata=cv.resize(imgdata,dsize=(275,200,3),interpolation=cv.INTER_CUBIC)
        #imgdata = imgdata.swapaxes(0,1)
        imgdata = cv.resize(imgdata,dsize=(200,275),interpolation=cv.INTER_CUBIC)
        #get data in correct format
        imgdata = imgdata.swapaxes(0,2)
        
        #img = Image.fromarray(imgdata)
        return imgdata/255.0

    def remember(self,state,action,reward,next_state,done):
        if TEST:
            return
        self.memory.append((state,action,reward,next_state,done))
    
    def train_short_memory(self,state,action,reward,next_state,done):
        if TEST:
            return
        self.trainer.train_step(state,action,reward,next_state,done)

    def train_long_memory(self):
        if TEST:
            return
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
        self.trainer.train_step(np.array(states),actions,rewards,np.array(next_states),dones)
    
    
    def get_action(self,state):
        #random moves: tradeoff exporation/exploitation
        #self.epsilon = STARTING_EPSILON - self.n_games
        #get empty list of 0s to replace with move
        move = [0,0,0,0,0,0,0,0,0]
        
        state = np.array(state)
        #Could change what goes into the randint max
        if self.random_moves_remaining > 0:
            printt('KEEP RANDOM')
            move[self.random_move_index] = 1
            self.random_moves_remaining -=1 
        elif random.randrange(0,100)/100 < self.epsilon:
            printt('RANDOM')
            if(ALLOW_DIAGONALS):
                move_index = random.randint(0,8)
            else:
                move_index = random.randint(0,4)
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




