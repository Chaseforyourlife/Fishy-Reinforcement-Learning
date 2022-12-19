import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from variables import *
import cv2 as cv
import numpy as np
import os

TELEMETRY = False

def printt(*strings):
    if TELEMETRY:
        for string in strings:
            print(string)


def optuna_get_model(trial,is_actor):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", OPTUNA_MODEL[0][0],OPTUNA_MODEL[0][1])
    layers = []
    in_features = INPUT_SIZE
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), OPTUNA_MODEL[1][0],OPTUNA_MODEL[1][1])
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        if OPTUNA_DROPOUT:
            p = trial.suggest_float("dropout_l{}".format(i),OPTUNA_DROPOUT[0],OPTUNA_DROPOUT[1])
            layers.append(nn.Dropout(p))
        in_features = out_features
    
    if is_actor:
        layers.append(nn.Linear(in_features, OUTPUT_SIZE))
        layers.append(nn.LogSoftmax(dim=1))
    else:
        layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)

class ActorNetwork(nn.Module):
    def __init__(self,sizes,trial=None):
        super(ActorNetwork,self).__init__()
        self.checkpoint_file = os.path.join('model','ppo_actor_model')
      
        
        if GRAYSCALE:
            self.paper_conv1 = nn.Conv2d(1+PREV_FRAME_NUMBER, 16, 8, stride=4, padding=1)
        elif not GRAYSCALE:
            self.paper_conv1 = nn.Conv2d((3 if not CENTER_AGENT else 4)*(1+PREV_FRAME_NUMBER), 16, 8, stride=4, padding=1)
        self.paper_conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.paper_flat = nn.Flatten()
        self.paper_lin1 = nn.Linear(23328,256)
        self.paper_lin2 = nn.Linear(256,SIZES[-1])
        




        if GRAYSCALE:
            self.conv1 = nn.Conv2d(1+PREV_FRAME_NUMBER, 8, 3, stride=1, padding=1)
        elif not GRAYSCALE:
            self.conv1 = nn.Conv2d((3 if not CENTER_AGENT else 4) *(1+PREV_FRAME_NUMBER), 8, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(3, 3)
        
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3, 3)
        #self.conv4 = nn.Conv2d(8, 8, 2, stride=1, padding=1)
        self.flat = nn.Flatten()        #self.lin1 = nn.Linear(16*(1+PREV_FRAME_NUMBER),16)
        self.lin0 = nn.Linear(10000,1500)
        self.lin1 = nn.Linear(1500,1000)
        self.lin2 = nn.Linear(1000,250)
        self.relu = nn.ReLU()
        self.lin3 = nn.Linear(250, SIZES[-1])
        self.soft = nn.Softmax(dim=-1)

        
        if OPTUNA and OPTUNA_MODEL:
            self.actor = optuna_get_model(trial,is_actor=True)
        lr = LEARNING_RATE
        if OPTUNA and OPTUNA_LR:
            lr = trial.suggest_float("lr",OPTUNA_LR[0],OPTUNA_LR[1],log=True)
        if OPTUNA and OPTUNA_OPTIM:
            optimizer_name  = trial.suggest_categorical("optimizer",OPTUNA_OPTIM)
            self.optimizer=getattr(optim,optimizer_name)(self.parameters(),lr=lr)
        else:
            self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self,value):
        #value is state
        '''
        printt(value.shape)
        value = self.conv1(value)
        printt(value.shape)
        if SHOW_CONV1:
            
            instate = value.clone().to('cpu').detach().numpy()[0]
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
        value = self.pool1(value)
        printt(value.shape)
        if SHOW_POOL1:
            instate = value.clone().to('cpu').detach().numpy()[0]

            #combined = instate.swapaxes(0,2)
            #cv.imshow(f'screen combined',cv.resize(combined,dsize=(SRN_SZE,SRN_SZE),interpolation=0))
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
        value = self.conv2(value)
        printt(value.shape,'after conv2')
        if SHOW_CONV2:
            instate = value.clone().to('cpu').detach().numpy()[0]
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
        value = self.pool2(value)
        printt(value.shape,'after pool2')
        if SHOW_POOL2:
            instate = value.clone().to('cpu').detach().numpy()[0]
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
        #value = self.conv3(value)
        printt(value.shape,'after conv3')
        
        if SHOW_CONV3:
            instate = value.clone().to('cpu').detach().numpy()[0]
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
        
        #value = self.pool3(value)
        printt(value.shape,'after pool3')
        if SHOW_POOL3:
            instate = value.clone().to('cpu').detach().numpy()[0]
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
          
        #value = self.conv4(value)
        printt(value.shape,'after conv4')
        if SHOW_CONV4:
            instate = value.clone().to('cpu').detach().numpy()[0]
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
        
        value = self.flat(value)
        value = self.lin0(value)
        value = self.relu(value)
        value = self.lin1(value)
        value = self.relu(value)
        value = self.lin2(value)
        value = self.relu(value)
        value = self.lin3(value)
        value = self.soft(value)
        #dist = self.actor(state)
        '''
        value = self.paper_conv1(value)
        value = self.relu(value)
        if SHOW_CONV1:
            
            instate = value.clone().to('cpu').detach().numpy()[0]
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
        value = self.paper_conv2(value)
        value = self.relu(value)
        if SHOW_CONV2:
            
            instate = value.clone().to('cpu').detach().numpy()[0]
            for i in range(len(instate)):
                cv.imshow(f'screen{i}',cv.resize(instate[i],dsize=(SRN_SZE,SRN_SZE),interpolation=0))
        value = self.flat(value)
        value = self.paper_lin1(value)
        value = self.relu(value)
        value = self.paper_lin2(value)
        value = self.soft(value)



        dist=Categorical(value)
        return dist
    def state_to_screen(self,state):
    
        screens=[]
        for frame_num,frame in enumerate(state):
            '''
            RGBscreens = [np.zeros(shape=(window_size[1],window_size[0],1),dtype='bool') for _ in range(3)]
            for x1,y1,x2,y2,fish_type in frame:
                RGBscreens[fish_type][y1:y2,x1:x2] = 1
          
            # GRAY SCALE IMAGE
            screen = np.vstack(RGBscreens)
            '''
            if not CENTER_AGENT:
                screen = np.zeros(shape=(window_size[1],window_size[0],3),dtype='bool')
            elif CENTER_AGENT:
                screen = np.zeros(shape=(window_size[1]*2,window_size[0]*2,4),dtype='bool')
            # get fishy coords for centering
            for x1,y1,x2,y2,fish_type in frame:
                if fish_type == 2:
                    fishy_coords = (x1,y1,x2,y2)
                fishy_center_w = (fishy_coords[0]+fishy_coords[2])/2
                fishy_center_h = (fishy_coords[1]+fishy_coords[3])/2
                offset_w= int((window_size[0]/2) + (window_size[0]/2-fishy_center_w))
                offset_h= int((window_size[1]/2) + (window_size[1]/2-fishy_center_h))
            # write fish to screen as boxes
            for x1,y1,x2,y2,fish_type in frame:
                if not CENTER_AGENT:
                    screen[y1:y2,x1:x2,fish_type] = 1
                elif CENTER_AGENT:
                    #already integers
                    screen[y1+offset_h:y2+offset_h,x1+offset_w:x2+offset_w,fish_type] = 1
            
            
            if GRAYSCALE:
                screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
            if CENTER_AGENT and GRAYSCALE:
                centered_screen = np.ones(shape=(window_size[1]*2,window_size[0]*2))
                
                centered_screen[offset_h:offset_h+window_size[1] , offset_w:offset_w+window_size[0]] = screen
                screen = centered_screen
                
            elif CENTER_AGENT and not GRAYSCALE:
                centered_screen = np.ones(shape=(window_size[1]*2,window_size[0]*2))
                fishy_center_w = (fishy_coords[0]+fishy_coords[2])/2
                fishy_center_h = (fishy_coords[1]+fishy_coords[3])/2
                offset_w= int((window_size[0]/2) + (window_size[0]/2-fishy_center_w))
                offset_h= int((window_size[1]/2) + (window_size[1]/2-fishy_center_h))
                centered_screen[offset_h:offset_h+window_size[1] , offset_w:offset_w+window_size[0]] = 0
                screen[:,:,3]=centered_screen
            screen = screen.astype('float32')
            screen = cv.resize(screen,dsize=window_resize)
            if SHOW_STATE_SCREEN and frame_num==0:
                showscreen=screen
                #showscreen = cv.resize(screen,dsize=(200,200))
                #showscreen = screen.astype('float32')
                #showscreen = cv.cvtColor(showscreen, cv.COLOR_BGR2GRAY)
                cv.imshow('frame',showscreen[:,:,:3])
                cv.waitKey(1)
            
            if GRAYSCALE:
                screen = screen.swapaxes(0,1)
            elif not GRAYSCALE:
                screen = screen.swapaxes(0,2)
            #add this for grayscale
            


          
            screens.append(screen)
        if not GRAYSCALE:
            screen = np.concatenate(screens,axis=0)
        elif GRAYSCALE:
            screen = np.stack(screens,axis=0)
       

        return screen
    

    def save_checkpoint(self):
        torch.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self,sizes,trial=None):
        super(CriticNetwork,self).__init__()
        self.checkpoint_file = os.path.join('model','ppo_critic_model')
        if GRAYSCALE:
            self.paper_conv1 = nn.Conv2d(1+PREV_FRAME_NUMBER, 16, 8, stride=4, padding=1)
        elif not GRAYSCALE:
            self.paper_conv1 = nn.Conv2d((3 if not CENTER_AGENT else 4)*(1+PREV_FRAME_NUMBER), 16, 8, stride=4, padding=1)
        self.paper_conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.paper_flat = nn.Flatten()
        self.paper_lin1 = nn.Linear(23328,256)
        self.paper_lin2 = nn.Linear(256,1)
        
        if GRAYSCALE:
            self.conv1 = nn.Conv2d(1+PREV_FRAME_NUMBER, 8, 3, stride=1, padding=1)
        elif not GRAYSCALE:
            self.conv1 = nn.Conv2d((3 if not CENTER_AGENT else 4)*(1+PREV_FRAME_NUMBER),8, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(3, 3)
        
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.conv4 = nn.Conv2d(8, 8, 2, stride=1, padding=1)
        self.flat = nn.Flatten()
        #self.lin1 = nn.Linear(16*(1+PREV_FRAME_NUMBER),16)
        self.lin0 = nn.Linear(10000,1500)
        self.lin1 = nn.Linear(1500,1000)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(1000,250)
        self.lin3 = nn.Linear(250, 1)
       
        if OPTUNA and OPTUNA_MODEL:
            self.critic = optuna_get_model(trial,is_actor=False)
        self.optimzier = optim.Adam(self.parameters(),lr=LEARNING_RATE)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self,value):
        value = self.paper_conv1(value)
        value = self.relu(value)
        value = self.paper_conv2(value)
        value = self.relu(value)
        value = self.flat(value)
        value = self.paper_lin1(value)
        value = self.relu(value)
        value = self.paper_lin2(value)
        return value
        '''
        #sprint(value.shape)
        value = self.conv1(value)
        value = self.pool1(value)
        value = self.conv2(value)
        value = self.pool2(value)
        #value = self.conv3(value)
        
        #value = self.pool3(value)
        #value = self.conv4(value)
        
        value = self.flat(value)
        value = self.lin0(value)
        value = self.relu(value)
        value = self.lin1(value)
        value = self.relu(value)
        value = self.lin2(value)
        value = self.relu(value)
        value = self.lin3(value)
        '''
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))












