import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from classes import Fishy,School
from variables import *
pygame.init()


fishy_background = pygame.image.load('../static/images/fishy-background.png')
screen = pygame.display.set_mode(window_size)
FPS = 30
class FishyGameAI:

    def __init__(self):
        self.window_size = window_size
        self.frame_number = 0
        # init display
        self.display = pygame.display.set_mode(window_size)
        pygame.display.set_caption('Fishy')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.main_fishy = Fishy()
        self.main_school = School()
        self.main_school.update()
        self.time_alive = 0
       
        self.win = False
        self.done = False



    def play_step(self, move):
        
        self.frame_number += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.main_school.update()
        #move fish_list
        self.main_school.move()
        
        #handle move
        self.main_fishy.handle_move(move)

        if self.main_fishy.fish_eaten >= MAX_FISH_SIZE+4:
            self.win = True
            self.done = True
        if self.main_fishy.alive == False:
            self.done = True
        if self.time_alive > TIME_MAX:
            self.done = True

        #handle main_fishy movement
        flipped,stopped = self.main_fishy.move()
        #check if fishy collided with any fish in the main_school
        fish_eaten = self.main_fishy.check_collide(self.main_school)

        

        reward = self.calculate_reward(self.main_fishy,self.main_school,fish_eaten,self.win,flipped,stopped)
        #print(reward)
        self._update_ui()
        self.time_alive += 1/FPS
        #self.clock.tick(FPS)

        return reward, self.done, self.main_fishy.fish_eaten


   


    def _update_ui(self):
        screen.blit(fishy_background,(0,0))
        #draw every fish in the main_school
        self.main_school.draw(screen)
        #draw fishy on the screen
        self.main_fishy.draw(screen)
        pygame.display.update()

    def calculate_reward(self,fishy,school,fish_eaten,win,flipped,stopped):
        reward = 0
        #get reward based on distance from fish
        for fish in school.fish_list:
            
            temp_reward = 25/max(1,((abs(fishy.x-fish.x)+abs(fishy.y-fish.y))))
            if fish.fish_eaten>fishy.fish_eaten:
                #reward+=-1*temp_reward
                pass
            else:
                #reward+=temp_reward
                pass
        #print(flipped)

        if flipped:
            reward -=10
            pass
        if stopped:
            reward -=10
            pass
        if fishy.alive:
            #reward -= 1
            pass
        else:
            #reward -= 50
            pass
        reward += fish_eaten*50
        if win:
            #reward += 1000
            pass
        #print('REWARD',reward)
        return reward
    