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
        temp_reward = 2.5/max(1,math.sqrt((abs(fishy.x-fish.x)**2+abs(fishy.y-fish.y)**2)))
        if fish.fish_eaten>=fishy.fish_eaten:
            if REWARD_PROXIMITY:
                reward+=-1*temp_reward
            pass
        else:
            if REWARD_PROXIMITY:
                reward+=temp_reward
            pass
    #print(flipped)

    if flipped:
        reward -=.2
        pass
    if stopped:
        reward -=.2
        pass
    if fishy.alive:
        #reward -= 1
        pass
    else:
        if REWARD_EAT:
            reward -=1
    if REWARD_EAT:
        #reward += fish_eaten*1
        if fish_eaten:
            reward+=1
    if win:
        #reward += 1000
        pass
    #print('REWARD',reward)
    return reward


class PPOMemory:
    def __init__(self,batch_size,trial=None):
        self.states=[]
        self.probs=[]
        self.vals=[]
        self.actions=[]
        self.rewards=[]
        self.dones=[]
        if OPTUNA and OPTUNA_BATCH_SIZE:
            self.batch_size=trial.suggest_int("batch_size",OPTUNA_BATCH_SIZE[0],OPTUNA_BATCH_SIZE[1],log=True)
        else:
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
    def __init__(self,trial=None):
        if OPTUNA and OPTUNA_GAMMA:
            self.gamma= trial.suggest_float("gamma",OPTUNA_GAMMA[0],OPTUNA_GAMMA[1],log=True)
        else:    
            self.gamma=GAMMA # discount rate
        self.policy_clip=POLICY_CLIP
        if OPTUNA and OPTUNA_EPOCHS:
            self.n_epochs=trial.suggest_int("epochs",OPTUNA_EPOCHS[0],OPTUNA_EPOCHS[1],log=True)
        else:
            self.n_epochs=NUM_EPOCHS
        self.gae_lambda=GAE_LAMBDA
        self.n_games = 0
        self.trial = trial

        self.actor=ActorNetwork(SIZES,trial)
        self.critic=CriticNetwork(SIZES,trial)
        self.memory=PPOMemory(BATCH_SIZE,trial)


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
        #distances from sides
        game_state.append(fishy.x/window_size[0]) #distance from left 
        game_state.append((window_size[0]-(fishy.x+fishy.width))/window_size[0]) #distance from right
        game_state.append(fishy.y/window_size[1]) #distance from up
        game_state.append((window_size[1]-(fishy.y+fishy.height))/window_size[1]) #distance from down 
        #Input Layer Data, input about fishy
        #game_state.append(fishy.x/window_size[0]) #x1
        #game_state.append(fishy.y/window_size[1]) #y1
        #game_state.append((fishy.x + fishy.width)/window_size[0]) #x2
        #game_state.append((fishy.y + fishy.height)/window_size[1]) #y2
        #game_state.append(fishy.x_speed/10)
        #game_state.append(fishy.y_speed/10)
        #Add data for all fish
        for fish in school.fish_list:
            #RELATIVE POSITIONS
            #game_state.append((fish.x-fishy.x)/window_size[0])
            #game_state.append((fish.y-fishy.y)/window_size[1])
            
            x_dis = None
            if fish.x > fishy.x+fishy.width:
                #right of fishy
                x_dis = fish.x - (fishy.x+fishy.width)
            elif fish.x+fish.width < fishy.x:
                #left of fishy (negative)
                x_dis = (fish.x+fish.width) - fishy.x
            else:
                x_dis = 0.0
            y_dis = None
            if fish.y > fishy.y+fishy.height:
                #below fishy
                y_dis = fish.y - (fishy.y+fishy.height) 
            elif fish.y+fish.height < fishy.y:
                #above fishy (negative)
                y_dis = (fish.y+fish.height) - fishy.y
            else:
                y_dis = 0.0
            game_state.append(x_dis/window_size[0])
            game_state.append(y_dis/window_size[1])
            #^RELATIVE POSITIONS
            
            #ABSOLUTE POSITIONS
            #game_state.append(fish.x/window_size[0]) #x1
            #game_state.append(fish.y/window_size[1]) #y1
            #game_state.append((fish.x + fish.width)/window_size[0]) #x2
            #game_state.append((fish.y + fish.height)/window_size[1]) #y2
        
            game_state.append(abs(fish.x_speed/10) if fish.x_speed>0 else 0.0)
            game_state.append(abs(fish.x_speed/10) if fish.x_speed<0 else 0.0)
            # need 2 nodes, can't use one node because seperate actions need to occur for both smaller and bigger
            game_state.append(float(fish.fish_eaten>fishy.fish_eaten)) #is_bigger
            game_state.append(float(fish.fish_eaten<=fishy.fish_eaten)) #is_smaller
            #game_state.append(fish.width/window_size[0])
            #game_state.append(fish.height/window_size[1])
        #print(game_state)
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
        #print(game_state)
        return np.array(game_state,dtype=float)

    def remember(self,state,action,prob,val,reward,done):
        if TEST:
            return
        self.memory.store_memory(state,action,prob,val,reward,done)
    def save_models(self):
        if SAVE_MODEL:
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
    
