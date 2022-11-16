import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from variables import *

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
        self.actor = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(7488,1000),
            nn.Linear(1000,250),
            nn.Linear(250, SIZES[-1]),
            nn.Softmax(dim=-1)
        )
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
    
    def forward(self,state):
        dist = self.actor(state)
        dist=Categorical(dist)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self,sizes,trial=None):
        super(CriticNetwork,self).__init__()
        self.checkpoint_file = os.path.join('model','ppo_critic_model')
        self.critic = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(7488,1000),
            nn.Linear(1000,250),
            nn.Linear(250, 1)
        )
        if OPTUNA and OPTUNA_MODEL:
            self.critic = optuna_get_model(trial,is_actor=False)
        self.optimzier = optim.Adam(self.parameters(),lr=LEARNING_RATE)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self,state):
        value = self.critic(state)
        #print(self.critic)
        #print(value)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))














'''
class Linear_QNet(nn.Module):
    def __init__(self,sizes):
        super().__init__()
        self.linear1 = nn.Linear(sizes[0],sizes[1])
        self.linear2 = nn.Linear(sizes[1],sizes[-1])
        #self.linear2 = nn.Linear(hidden_size,output_size)
        #self.linear3 = nn.Linear(sizes[2],sizes[-1])
        #self.linear4 = nn.Linear(sizes[2],sizes[4])
        self.load()
    def forward(self,x):
        #x=x.to(DEVICE)
        printt('IN:',x)
        x = F.relu(self.linear1(x))
        #printt('1',x)
        #x = self.linear1(x)
        #x = F.relu(self.linear2(x))
        printt('2',x)
        x = self.linear2(x)
        #x = F.relu(self.linear3(x))
        #x= self.linear4(x)
        #print('OUT:',x)
        return x

    def save(self,optimizer,file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save({'model_state_dict':self.state_dict(),'optimizer_state_dict':optimizer.state_dict()},file_name)
    def load(self,file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name)
            self.load_state_dict(checkpoint['model_state_dict'])
           
'''
'''
class QTrainer:
    def __init__(self,model,learning_rate,gamma):
        self.model = model#.to(DEVICE)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.load()

    def load(self,file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train_step(self,state,action,reward,next_state,done):
        #print(state,done)
        #print("TRAINSTEP")
        #foo =zip(state,action,reward,done)
        #for state,action,reward,done in foo:
            #print('\nstate',state,'\naction',action,'\nreward',reward,'\ndone',done)
        #   pass
            
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.float)
        reward = torch.tensor(reward,dtype=torch.float)
        #print(state)
        if len(state.shape)==1: # reshape tensors if there only one state is being trained on
            print('SHAPE')
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        ##Predicted Q values with the current state
        pred = self.model(state)

        target = pred.clone()
        #print('HIIIIIIII')
        #print(len(done))
        #print(state)
        #print(done)
        Q = reward[0]
        for index in range(0,len(done)):
            #print('TRAINNIININ')
            printt(action[index])
            printt(reward[index])
            #if done[index-1]:
            #    Q=reward[index]
            #Q=reward[index]
            # elif not done[index-1]:
            #if not done[index-1]:
                #printt('not done')
            #    Q = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            if not done[index]:
                #print('this',torch.max(self.model(next_state[index])))
                Q = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            else:
                Q = reward[index]*((1+self.gamma) if reward[index]==1 else 1)
            #print('reward',reward[index])
            #print('q_new',Q)
            #print('pred',pred[index])
            #print('Q',Q)
            target[index][torch.argmax(action[index]).item()] = Q
        ## Q_new reward + gamma * max(next_predicted Q value) -> only do this if not done
        #pred.clone()
        #preds[torch.argmax(action)] = Q

        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        #back propogation
        loss.backward()

        self.optimizer.step()
'''