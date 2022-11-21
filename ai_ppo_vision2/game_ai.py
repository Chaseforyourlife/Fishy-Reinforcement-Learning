import re
import torch
import random
import numpy as np
from collections import deque
from model import ActorNetwork, CriticNetwork#, QTrainer
from variables import *
import math
from itertools import islice
import cv2 as cv
import pygame
from PIL import Image
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
    for fish in school.fish_list:
        over=False
        lined_up=False
        #Right and Left collision 
        if(fishy.x < fish.x + fish.width and fishy.x + fishy.width > fish.x):
            over = True
        #Top and bottom collision
        if (fishy.y < fish.y + fish.height and fishy.y + fishy.height > fish.y):
            lined_up = True 
        if over and lined_up and fishy.fish_eaten < fish.fish_eaten and fish.alive == True:
            reward-=1
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
        n_states=len(self.states)-1 #-1 because we are discarding the last state
        batch_start = np.arange(0,n_states,self.batch_size)
        printt(batch_start)
        indicies = np.arange(n_states,dtype=np.int64)
        np.random.shuffle(indicies)
        batches = [indicies[i:min(i+self.batch_size,n_states)] for i in batch_start]
        printt('GenerateBatches:',batches)
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



        self.random_moves_remaining = 0
        self.random_move_index = None
        
    def get_state(self,fishy,school):
        int_convert = lambda x:int(max(x,0))
        fishy_coords = tuple(map(int_convert,(fishy.x,fishy.y,fishy.x+fishy.width,fishy.y+fishy.height,2)))
        school_coords = [fishy_coords]
        for fish in school.fish_list:
            fish_coords = tuple(map(int_convert,(fish.x,fish.y,fish.x+fish.width,fish.y+fish.height,1 if fishy.fish_eaten >= fish.fish_eaten else 0)))
            school_coords.append(fish_coords)
        game_state = [school_coords]
        for i in range(PREV_FRAME_NUMBER):
            if len(self.memory.states)>i:
                prev_state = self.memory.states[-(i+1)][0]
                game_state.append(prev_state)
            else:
                game_state.append(game_state[0])
        return game_state
        '''
        screen = np.zeros(shape=(window_size[1],window_size[0],3),dtype='bool')
        screen[int(max(fishy.y,0)):int(max(fishy.y+fishy.height,0)),int(max(fishy.x,0)):int(max(fishy.x+fishy.width,0)),2] = 1
        for fish in school.fish_list:
            screen[max(fish.y,0):max(fish.y+fish.height,0),max(fish.x,0):max(fish.x+fish.width,0),1 if fishy.fish_eaten >= fish.fish_eaten else 0] = 1 
        if SHOW_STATE_SCREEN:
            showscreen = screen.astype('float')
            cv.imshow('frame',showscreen[:,:,::-1])
            cv.waitKey(1)
        
        screen = screen.swapaxes(0,2)
        
        screens = [screen]
        #imgdata = imgdata.swapaxes(0,1)
        for i in range(PREV_FRAME_NUMBER):
            if len(self.memory.states)>i:
                prev_screen = self.memory.states[-(i+1)][0:3]
                screens.append(prev_screen)
            else:
                screens.append(screen)
        screen = np.concatenate(screens)
        return screen
        '''
        
    

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
    
        observation_screen = self.actor.state_to_screen(observation)
        
        state = torch.tensor(np.array([observation_screen]),dtype=torch.float)
        
        state =state.to(self.actor.device)
        #print(f'STATe {state}')
        dist=self.actor(state)
        #print('DIST',dist)
        

        value=self.critic(state)
        #print('VALUE',value)
        #print(dist.probs)
    
        
        action=dist.sample()
        

        #print('ACTION',action)
        #print(dist.log_prob(action))
        probs=torch.squeeze(dist.log_prob(action)).item()
        action=torch.squeeze(action).item()
        
        
        #move = [0,0,0,0,0,0,0,0,0]
        value=torch.squeeze(value).item()
        
        #move[action]=1


        return action,probs,value
    
    def learn(self):
        for _ in range(self.n_epochs):
            printt('Starting Epoch')
            state_arr,action_arr,old_probs_arr,vals_arr,reward_arr,done_arr,batches=self.memory.generate_batches()
            
            #NVM: originally: both longer by 1
            values = vals_arr
            advantage=np.zeros(len(reward_arr),dtype=np.float32)
            printt('values:',values)
            printt(len(values))
            #Loop through total timesteps
            #k_values = set()
            #k_plus_one_values = set()
            #reward_arr=[]
            #printt('reward_arr[0]:',reward_arr[0])
            #printt('reward_arr[-1]:',reward_arr[-1])
            for t in range(len(reward_arr)-1):
                #printt(reward_arr[t])
                discount=1
                a_t = 0
                timesteps=0
                for k in range(t,len(reward_arr)-1):
                    #k_values.add(k)
                    #printt(reward_arr[k])
                    if timesteps >=TIMESTEPS_PER_ITERATION:
                        break
                    
                    a_t += discount*(reward_arr[k]+self.gamma*values[k+1]*(1-int(done_arr[k]))-values[k])
                    discount *= self.gamma*self.gae_lambda
                    timesteps+=1
                    #if t==len(reward_arr)-4:
                        #print(reward_arr[k])
                        #print(a_t)
                advantage[t] = a_t
            #printt(k_values)
            #printt(len(advantage))
            #printt(advantage)
            #printt(advantage[len(reward_arr)-1])
           
            print()
            print('LAST REWARDS:',reward_arr[-10:])
            print('LAST ADVANTAGES:',advantage[-10:])
            print('LAST VALUES:',values[-10:])
            
            ###NORMALIZE REWARDS???
            #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            ###^Not in original
            advantage = torch.tensor(advantage.copy()).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)
            printt('Batches Ready')
            printt('Batches:',batches)
            for count,batch in enumerate(batches):
                
                new_state_arr = []
                states = state_arr[batch]
                for state in states:
                    new_state_arr.append(self.actor.state_to_screen(state))

                printt(f'Batch {count}')
                printt(f'BATCH: {batch}')
                #states = torch.tensor(state_arr[batch],dtype=torch.float).to(self.actor.device)
                states = torch.tensor(np.array(new_state_arr),dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                #printt('states',states)
                printt('old_probs',old_probs)
                printt('actions',actions)
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

                printt(f'ACTOR LOSS: {actor_loss}')
                printt(f'CRITIC LOSS: {critic_loss}')


                #total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimzier.zero_grad()
                #total_loss.backward()
                actor_loss.backward()
                critic_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimzier.step()
            
        self.memory.clear_memory()
    
