import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from variables import *

import os

TELEMETRY = False

def printt(*strings):
    if TELEMETRY:
        for string in strings:
            print(string)

class Linear_QNet(nn.Module):
    def __init__(self,output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.load()
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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
           

class QTrainer:
    def __init__(self,model,learning_rate,gamma):
        self.model = model
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
        
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.float)
        reward = torch.tensor(reward,dtype=torch.float)

        if len(state.shape)==1: # reshape tensors if there only one state is being trained on
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        ##Predicted Q values with the current state
        pred = self.model(state)

        target = pred.clone()
        for index in range(len(done)):
            printt(action[index])
            printt(reward[index])
            Q_new = reward[index]
            #if not done[index]:
                #printt('not done')
                #Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            target[index][torch.argmax(action[index]).item()] = Q_new
        ## Q_new reward + gamma * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        #back propogation
        loss.backward()

        self.optimizer.step()