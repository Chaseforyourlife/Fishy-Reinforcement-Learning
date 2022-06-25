import pygame
import random
import game_ai as GAI

#VARIABLES#
window_size = (550,400)
LEFT_IMAGES = {
  'orange':pygame.image.load('static/images/fishy_left.png'),
  'pink':pygame.image.load('static/images/fishy_left_pink.png'),
  'purple':pygame.image.load('static/images/fishy_left_purple.png'),
  'blue':pygame.image.load('static/images/fishy_left_blue.png')
}
RIGHT_IMAGES = {
  'orange':pygame.image.load('static/images/fishy_right.png'),
  'pink':pygame.image.load('static/images/fishy_right_pink.png'),
  'purple':pygame.image.load('static/images/fishy_right_purple.png'),
  'blue':pygame.image.load('static/images/fishy_right_blue.png')
}

screen = pygame.display.set_mode(window_size)
FPS = 30
MAX_FISH = 8
MAX_FISH_SPEED = 6
MIN_FISH_SPEED = 2

class Fishy:
  ##Init function
  def __init__(self):
    #positional variables
    self.width = 40
    self.height = 8 
    self.x = 225
    self.y = 200
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
    #change speed and direction based on keys pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
      self.handle_move('L')
    if keys[pygame.K_RIGHT]:
      self.handle_move('R')
    if keys[pygame.K_UP]:
      self.handle_move('U')
    if keys[pygame.K_DOWN]:
      self.handle_move('D')
  def handle_move(self,direction=None):
    if direction == 'L':
      self.x_speed -= .25
      self.direction = -1
    elif direction == 'R':
      self.x_speed += .25
      self.direction = 1  
    elif direction == 'U':
      self.y_speed -= .25
    elif direction == 'D':
      self.y_speed += .25
  ##Handle everything related to fishy moving
  def move(self):
    #if speed is below speed_change, set speed to 0
    if abs(self.x_speed) < self.x_speed_change:
      self.x_speed = 0 
    if abs(self.y_speed) < self.y_speed_change:
      self.y_speed = 0 
    #check if fish is at top of bottom of screen before changing position
    if self.y <  0:
      self.y = 0
      self.y_speed = 0
    elif self.y + self.height > window_size[1]:
      self.y = window_size[1] - self.height
      self.y_speed = 0
    #check if fish is past edge of screen before changing position
    if self.x + self.width/2 < 0:
      self.x = window_size[0] - self.width/2
    elif self.x + self.width/2 > window_size[0]:
      self.x = 0 - self.width/2
    #decrease speed overtime and set max speed
    self.x_speed -= self.x_speed/(self.max_x_speed/self.x_speed_change)
    self.y_speed -= self.y_speed/(self.max_y_speed/self.y_speed_change)
    #move by amount of speed
    self.x += self.x_speed
    self.y += self.y_speed
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
    self.move()
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
      fish_eaten=random.randint(-35,150)
      #Generate fish
      width = 40 + int(fish_eaten*.2 * 5)
      height = 8 + int(fish_eaten*.2 * 1)
      direction = 1 if random.randint(0,1) else -1
      x_speed = random.randrange(MIN_FISH_SPEED,MAX_FISH_SPEED) * direction
      
      x = -width if direction == 1 else window_size[0]
      y = random.randint(0,window_size[1]-height)
      
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



