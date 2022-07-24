import torch
import random
import numpy as np
from collections import deque
from game import FishyGameAI
from model import Linear_QNet, QTrainer
from helper import plot
from variables import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(4, 16, 9) #256
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        game_state = []
        #Fishy x and y
        game_state.append((game.main_fishy.x+game.main_fishy.width/2)/game.window_size[0])
        game_state.append((game.main_fishy.y+game.main_fishy.height/2)/game.window_size[1])
        #fish x and y
       
        for fish in game.main_school.fish_list:
            game_state.append((fish.x+fish.width/2)/window_size[0])
            game_state.append((fish.y+fish.height/2)/window_size[1])
        return np.array(game_state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            return
            #mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0,0,0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 8)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = FishyGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

      

        ##Get AI Model Inference
        move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, move, reward, state_new, done)

        # remember
        agent.remember(state_old, move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()