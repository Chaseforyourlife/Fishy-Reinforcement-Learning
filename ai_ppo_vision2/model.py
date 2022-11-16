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
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(1536,250),
            nn.ReLU(),
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
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(1536,250),
            nn.ReLU(),
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












