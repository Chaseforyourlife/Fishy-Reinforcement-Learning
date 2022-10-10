import pygame
import random
from game_ai import *
import collections
from variables import *

#VARIABLES#
window_size = (550,400)
LEFT_IMAGES = {
  'orange':pygame.image.load('../static/images/fishy_left.png'),
  'pink':pygame.image.load('../static/images/fishy_left_pink.png'),
  'purple':pygame.image.load('../static/images/fishy_left_purple.png'),
  'blue':pygame.image.load('../static/images/fishy_left_blue.png')
}
RIGHT_IMAGES = {
  'orange':pygame.image.load('../static/images/fishy_right.png'),
  'pink':pygame.image.load('../static/images/fishy_right_pink.png'),
  'purple':pygame.image.load('../static/images/fishy_right_purple.png'),
  'blue':pygame.image.load('../static/images/fishy_right_blue.png')
}

screen = pygame.display.set_mode(window_size)
FPS = 30


class Fishy:
  ##Init function
  def __init__(self):
    #positional variables
    self.moves_counter = collections.Counter()
    self.width = 40
    self.height = 8 
    
    self.x = 225
    self.y = 200
    if RANDOM_START:
      self.x = random.randint(0,window_size[0]-self.width)
      self.y = random.randint(0,window_size[1]-self.height)
    self.direction = 1
    self.x_speed = 0
    self.y_speed = 0
    self.max_x_speed = 8
    self.max_y_speed = 8
    self.x_speed_change = .25
    self.y_speed_change = .25
    self.image_left = pygame.transform.scale(LEFT_IMAGES['orange'],(self.width,self.height)) 
    self.image_right = pygame.transform.scale(RIGHT_IMAGES['orange'],(self.width,self.height)) 
    #stat variables
    self.alive = True
    self.fish_eaten = 0
  ##Handle player key presses
  def handle_keys(self):
    move = [0,0,0,0,0,0,0,0]
    #change speed and direction based on keys pressed
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_LEFT] and keys[pygame.K_UP]:
      self.handle_move([0,0,0,0,0,1,0,0,0])
    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
      self.handle_move([0,0,0,0,0,0,1,0,0])
    elif keys[pygame.K_RIGHT] and keys[pygame.K_DOWN]:
      self.handle_move([0,0,0,0,0,0,0,1,0])
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
      self.handle_move([0,0,0,0,0,0,0,0,1])
    elif keys[pygame.K_LEFT]:
      self.handle_move([1,0,0,0,0,0,0,0,0])
    elif keys[pygame.K_RIGHT]:
      self.handle_move([0,1,0,0,0,0,0,0,0])
    elif keys[pygame.K_UP]:
      self.handle_move([0,0,1,0,0,0,0,0,0])
    elif keys[pygame.K_DOWN]:
      self.handle_move([0,0,0,1,0,0,0,0,0])
    else:
      self.handle_move([0,0,0,0,1,0,0,0,0])
    '''
    elif keys[pygame.K_LEFT]:
      self.handle_move([1,0,0,0,0])
    elif keys[pygame.K_RIGHT]:
      self.handle_move([0,1,0,0,0])
    elif keys[pygame.K_UP]:
      self.handle_move([0,0,1,0,0])
    elif keys[pygame.K_DOWN]:
      self.handle_move([0,0,0,1,0])
    else:
      self.handle_move([0,0,0,0,0])
    '''
  def handle_move(self,direction=None):
    if type(direction)==int:
      move = [0,0,0,0,0,0,0,0,0]
      move[direction]=1 
      direction = move
    self.moves_counter[direction.index(1)]+=1
    
    if direction[0]:
      self.x -= 1* SPEED
      self.direction = -1
    elif direction[1]:
      self.x += 1* SPEED
      self.direction = 1  
    elif direction[2]:
      self.y -= 1* SPEED
    elif direction[3]:
      self.y += 1* SPEED
    #THIS IS DON'T MOVE
    elif direction[4]:
      pass
    elif direction[5]:
      self.x -= .7* SPEED
      self.direction = -1
      self.y -= .7* SPEED
    elif direction[6]:
      self.x += .7* SPEED
      self.direction = 1  
      self.y -= .7* SPEED
    elif direction[7]:
      self.y += .7* SPEED
      self.x += .7* SPEED
      self.direction = 1
    elif direction[8]:
      self.y += .7* SPEED
      self.x -= .7* SPEED
      self.direction = -1
  ##Handle everything related to fishy moving
  def move(self):
    stopped = False
    flipped = False
    '''
    #if speed is below speed_change/2, set speed to 0
    if abs(self.x_speed) < self.x_speed_change/2:
      self.x_speed = 0 
    if abs(self.y_speed) < self.y_speed_change/2:
      self.y_speed = 0 
    '''
    #check if fish is at top of bottom of screen before changing position
    if self.y <=  0:
      self.y = 0
      self.y_speed = 0
      stopped = True
    elif self.y + self.height >= window_size[1]:
      self.y = window_size[1] - self.height
      self.y_speed = 0
      stopped = True
    #check if fish is past edge of screen before changing position
    
    if self.x + self.width/2 <= 0:
      #self.x = window_size[0] - self.width/2
      self.x = -self.width/2
      #remove later
      self.x_speed = 0
      flipped = True
  
    elif self.x + self.width/2 >= window_size[0]:
      #self.x = 0 - self.width/2
      self.x = window_size[0]-self.width/2
      flipped = True
      #remove later
      self.x_speed = 0
    #decrease speed overtime and set max speed
    self.x_speed -= self.x_speed/(self.max_x_speed/self.x_speed_change)
    self.y_speed -= self.y_speed/(self.max_y_speed/self.y_speed_change)
    #move by amount of speed
    self.x += self.x_speed
    self.y += self.y_speed
    return flipped,stopped

  ##Draw fishy image to screen
  def draw(self,screen):
    if self.alive:
      screen.blit(self.image_left if self.direction == -1 else self.image_right,(self.x,self.y))
  ##Called on a schedule to check if fishy collides with any fish
  def check_collide(self,school):
    fish_eaten = 0
    #if self.fish_eaten > 
    for fish in school.fish_list:
      over = False
      lined_up = False
      #Right and Left collision 
      if(self.x < fish.x + fish.width and self.x + self.width > fish.x):
        over = True
      #Top and bottom collision
      if (self.y < fish.y + fish.height and self.y + self.height > fish.y):
        lined_up = True 
      if over and lined_up:
        eaten = self.collide(fish)
        if eaten:
          fish_eaten +=1
    return fish_eaten
      
  def collide(self,other_fish):
    if other_fish.alive == False:
      return False
    elif self.fish_eaten >= other_fish.fish_eaten and other_fish.alive == True:
      other_fish.alive = False
      self.fish_eaten += 1
      self.width = 40 + int(self.fish_eaten*.2 * 5)
      self.height = 8 + int(self.fish_eaten*.2 * 1)
      self.image_left = pygame.transform.scale(LEFT_IMAGES['orange'],(self.width,self.height)) 
      self.image_right = pygame.transform.scale(RIGHT_IMAGES['orange'],(self.width,self.height))
      return True 
    else:
      self.alive = False
      return False

  

class Fish():
  ##Init function
  def __init__(self,width,height,x,y,direction,x_speed,fish_eaten,color):
    #positional variables
    self.width = width
    self.height = height
    self.x = x
    self.y = y
    self.direction = direction
    self.x_speed = x_speed
    #FIXME
    self.image_left = pygame.transform.scale(LEFT_IMAGES[color],(int(self.width),int(self.height))) 
    self.image_right = pygame.transform.scale(RIGHT_IMAGES[color],(int(self.width),int(self.height))) 
    #stat variables
    self.alive = True
    self.fish_eaten = fish_eaten
  
  def draw(self,screen):
    if self.alive:
      screen.blit(self.image_left if self.direction == -1 else self.image_right,(self.x,self.y))



class School():
  def __init__(self):
    self.fish_list = []
  
  def update(self):
    #move all existing fish
    #remove fish beyond boundaries 
    #add new fish randomly
    self.check_clear_fish()
    self.check_add_fish()

    

  def move(self):
    for fish in self.fish_list:
      fish.x += fish.x_speed
      if fish.x + fish.width < 0 and fish.direction == -1:
        fish.alive = False
      elif fish.x > window_size[0] and fish.direction == 1:
        fish.alive = False

  def check_clear_fish(self):
    for count,fish in enumerate(self.fish_list):
      if not fish.alive:
        self.fish_list.pop(count)

  def check_add_fish(self):
    #TODO add sizes to fish
    while len(self.fish_list) < MAX_FISH:
      fish_eaten=random.randint(MIN_FISH_SIZE,MAX_FISH_SIZE)
      #Generate fish
      width = 40 + int(fish_eaten*.2 * 5)
      height = 8 + int(fish_eaten*.2 * 1)
      direction = 1 if random.randint(0,1) else -1
      x_speed = random.randrange(MIN_FISH_SPEED,MAX_FISH_SPEED) * direction
      #x_speed = 0
      x = -width if direction == 1 else window_size[0]
      y = random.randint(0,int(window_size[1]-height))
      #x = 400
      if TRAINING_STATE=='TRAIN_X':
        x_speed = 0
        x = random.randint(0,window_size[0]-width)
        y = random.randint(200-X_TRAIN_Y_RANGE,200+X_TRAIN_Y_RANGE)
      elif TRAINING_STATE=='TRAIN_Y':
        x_speed = 0
        x = 225
        y = random.randint(0,window_size[1]-height)
      elif TRAINING_STATE=='TRAIN_XY':
        x_speed = 0
        if XY_TRAIN_RANGE > 0:
          x = random.randint(window_size[0]/2-XY_TRAIN_RANGE,window_size[0]/2+XY_TRAIN_RANGE)
          y = random.randint(window_size[1]/2-XY_TRAIN_RANGE,window_size[1]/2+XY_TRAIN_RANGE)
        else:
          x = random.randint(0,window_size[0]-width)
          y = random.randint(0,window_size[1]-height)
      #x = window_size[0]/2 - width/2
      #y = random.randint(int(-height/2),int(window_size[1]-height/2))
      #y = random.randint(0,int(window_size[1]-height))
      
      if fish_eaten > 120:
        color = 'blue'
      elif fish_eaten > 75:
        color = 'purple'
      elif fish_eaten > 30:
        color = 'pink'
      else:
        color = 'orange'
      new_fish = Fish(width,height,x,y,direction,x_speed,fish_eaten,color=color)
      self.fish_list.append(new_fish)

  def draw(self,screen):
    for fish in self.fish_list:
      fish.draw(screen)



