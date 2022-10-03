import re
import torch
import random
import numpy as np
from collections import deque
from model import ActorNetwork, CriticNetwork#, QTrainer
from variables import *
import math
from itertools import islice


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

def calculate_reward(fishy,school,fish_eaten,win,flipped,stopped,state_old):
    reward = 0
    #get reward based on distance from fish
    for fish in school.fish_list:
        #temp_reward = 
        temp_reward = 250/max(1,math.sqrt((abs(fishy.x-fish.x)**2+abs(fishy.y-fish.y)**2)))
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
            reward = -10
    if REWARD_EAT:
        #reward += fish_eaten*1
        if fish_eaten:
            reward=10
    if win:
        #reward += 1000
        pass
    printt('REWARD',reward)
    return reward


class PPOMemory:
    def __init__(self,batch_size):
        self.states=[]
        self.probs=[]
        self.vals=[]
        self.actions=[]
        self.rewards=[]
        self.dones=[]
        self.batch_size=batch_size
    def generate_batches(self):
        n_states=len(self.states)
        batch_start = np.arange(0,n_states,self.batch_size)
        indicies = np.arange(n_states,dtype=np.int64)
        np.random.shuffle(indicies)
        batches = [indicies[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
    def store_memory(self,state,action,prob,val,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
    def clear_memory(self):
        self.states=[]
        self.probs=[]
        self.vals=[]
        self.actions=[]
        self.rewards=[]
        self.dones=[]



class Agent:
    def __init__(self):
        self.gamma=GAMMA # discount rate
        self.policy_clip=POLICY_CLIP
        self.n_epochs=NUM_EPOCHS
        self.gae_lambda=GAE_LAMBDA
        self.n_games = 0
        self.epsilon = EPSILON  # randomness
        self.min_epsilon = MIN_EPSILON

        self.actor=ActorNetwork(SIZES)
        self.critic=CriticNetwork(SIZES)
        self.memory=PPOMemory(BATCH_SIZE)


        #self.memory = deque(maxlen=MAX_MEMORY)  # popleft() when exceeding max_memory
        #TODO: model,trainer
        #self.model = Linear_QNet(sizes=SIZES)
        #self.trainer = QTrainer(self.model,LEARNING_RATE,GAMMA)
        self.random_moves_remaining = 0
        self.random_move_index = None
        #self.model.load()
        #if LOAD_OPTIMIZER:
        #    self.trainer.load()
    def get_state(self,fishy,school):
        #####NORMALIZE INPUTS
        game_state = []
        
        #Input Layer Data
        game_state.append(fishy.x/window_size[0]) #x1
        game_state.append(fishy.y/window_size[1]) #y1
        game_state.append((fishy.x + fishy.width)/window_size[0]) #x2
        game_state.append((fishy.y + fishy.height)/window_size[1]) #y2
        #game_state.append(fishy.x_speed/10)
        #game_state.append(fishy.y_speed/10)
        #Add data for all fish
        for fish in school.fish_list:
            #game_state.append((fish.x-fishy.x)/window_size[0])
            #game_state.append((fish.y-fishy.y)/window_size[1])
            game_state.append(fish.x/window_size[0]) #x1
            game_state.append(fish.y/window_size[1]) #y1
            game_state.append((fish.x + fish.width)/window_size[0]) #x2
            game_state.append((fish.y + fish.height)/window_size[1]) #y2
            game_state.append(fish.x_speed/10)
            game_state.append(fish.fish_eaten>fishy.fish_eaten) #is_bigger
        #Add data for Map to Fishy
        #game_state.append(fishy.x) #distance from left 
        #game_state.append((window_size[0]-(fishy.x+fishy.width))/window_size[0]) #distance from right
        #game_state.append(fishy.y) #distance from up
        #game_state.append((window_size[1]-(fishy.y+fishy.height))/window_size[1]) #distance from down 
        '''
        #Fishy x and y
        game_state.append((fishy.x+fishy.width/2)/window_size[0])
        game_state.append((fishy.y+fishy.height/2)/window_size[1])
        #fish x and y
        for fish in school.fish_list:
            game_state.append((fish.x+fish.width/2)/window_size[0])
            game_state.append((fish.y+fish.height/2)/window_size[1])
        '''
        return np.array(game_state,dtype=float)

    def remember(self,state,action,prob,val,reward,done):
        if TEST:
            return
        self.memory.store_memory(state,action,prob,val,reward,done)
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    def get_action(self,observation):
        state = torch.tensor(np.array([observation]),dtype=torch.float).to(self.actor.device)
        dist=self.actor(state)
        #print('DIST',dist)
        value=self.critic(state)
        #print('VALUE',value)
        action=dist.sample()
        #print('ACTION',action)
        probs=torch.squeeze(dist.log_prob(action)).item()
        action=torch.squeeze(action).item()
        #move = [0,0,0,0,0,0,0,0,0]
        value=torch.squeeze(value).item()
        #move[action]=1


        return action,probs,value
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr,action_arr,old_probs_arr,vals_arr,reward_arr,done_arr,batches=self.memory.generate_batches()
            
            values = vals_arr
            advantage=np.zeros(len(reward_arr),dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount=1
                a_t = 0
                for k in range(t,len(reward_arr)-1):
                    a_t += discount*(reward_arr[k]+self.gamma*values[k+1]*(1-int(done_arr[k]))-values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch],dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio,1-self.policy_clip,1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs,weighted_clipped_probs).mean()
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimzier.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimzier.step()
            
            self.memory.clear_memory()
    '''
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
    '''



