import pygame
import random

#VARIABLES#
window_size = (550,400)
fishy_left_image = pygame.image.load('static/images/fishy_left.png')
fishy_right_image = pygame.image.load('static/images/fishy_right.png')
screen = pygame.display.set_mode(window_size)
FPS = 30
MAX_FISH = 4
MAX_FISH_SPEED = 10
MIN_FISH_SPEED = 3

class Fishy:
  ##Init function
  def __init__(self):
    #positional variables
    self.width = 38
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
    self.image_left = pygame.transform.scale(fishy_left_image,(self.width,self.height)) 
    self.image_right = pygame.transform.scale(fishy_right_image,(self.width,self.height)) 
    #stat variables
    self.alive = True
    self.fish_eaten = 0
  ##Handle everything related to fishy moving
  def move(self):

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
    #change speed and direction based on keys pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
      self.x_speed -= .25
      self.direction = -1
    if keys[pygame.K_RIGHT]:
      self.x_speed += .25
      self.direction = 1  
    if keys[pygame.K_UP]:
        self.y_speed -= .25
    if keys[pygame.K_DOWN]:
      self.y_speed += .25
    #if speed is below speed_change, set speed to 0
    if abs(self.x_speed) < self.x_speed_change:
      self.x_speed = 0 
    if abs(self.y_speed) < self.y_speed_change:
      self.y_speed = 0 
  ##Draw fishy image to screen
  def draw(self,screen):
    if self.alive:
      screen.blit(self.image_left if self.direction == -1 else self.image_right,(self.x,self.y))
  ##Called on a schedule to check if fishy collides with any fish
  def check_collide(self,school):

    #if self.fish_eaten > 
    
    for fish in school.fish_list:
      over = False
      lined_up = False
      #Right and Left collision 
      if(self.x <= fish.x + fish.width and self.x + self.width >= fish.x):
        over = True
      #Top and bottom collision
      if (self.y <= fish.y + fish.height and self.y + self.height >= fish.y):
        lined_up = True 
      print(over,lined_up)
      if over and lined_up:
        self.collide(fish)
      
  def collide(self,other_fish):
    if self.fish_eaten >= other_fish.fish_eaten and other_fish.alive == True:
      other_fish.alive = False
      self.fish_eaten += 1
    else:
      self.alive = False

  

class Fish():
  ##Init function
  def __init__(self,width,height,x,y,direction,x_speed,fish_eaten):
    #positional variables
    self.width = width
    self.height = height
    self.x = x
    self.y = y
    self.direction = direction
    self.x_speed = x_speed
    #FIXME
    self.image_left = pygame.transform.scale(fishy_left_image,(self.width,self.height)) 
    self.image_right = pygame.transform.scale(fishy_right_image,(self.width,self.height)) 
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
    if len(self.fish_list) < MAX_FISH:
      #Generate fish
      width = 40
      height = 8
      direction = 1 if random.randrange(0,1) else -1
      x_speed = random.randrange(MIN_FISH_SPEED,MAX_FISH_SPEED) * direction
      
      x = -width if direction == 1 else window_size[0]
      y = random.randrange(0,window_size[1]-height)

      #TODO FIXME
      fish_eaten=0
      
      new_fish = Fish(width,height,x,y,direction,x_speed,fish_eaten)
      self.fish_list.append(new_fish)

  def draw(self,screen):
    for fish in self.fish_list:
      fish.draw(screen)

#use this for fish
'''class Ship(pygame.sprite.Sprite):
    def __init__(self, image_file, speed, location):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location
You could then "activate" it like this:

ship = Ship("images\ship.png", [a, b])'''

def main():
  fishy_background = pygame.image.load('static/images/fishy-background.png')
  #screen.fill((255,255,255))
  
  #pygame.draw.rect(screen,(0,0,0),(225,200,40,20))
  clock = pygame.time.Clock()
  
  main_fishy = Fishy()
  main_school = School()
  
  running = True
  while running:
    #draw 
    screen.blit(fishy_background,(0,0))
    clock.tick(FPS)
    ##Initialize game start

    ###MAIN GAME LOOP AFTER START
    while main_fishy.alive == True:
      clock.tick(FPS)
      #draw background
      screen.blit(fishy_background,(0,0))
      #update fish_list
      main_school.update()
      #handle main_fishy movement
      main_fishy.move()
      #check if fishy collided with any fish in the main_school
      main_fishy.check_collide(main_school)
      #draw every fish in the main_school
      main_school.draw(screen)
      #draw fishy on the screen
      main_fishy.draw(screen)
      
      
      #TESTING PURPOSES
      main_fishy.fish_eaten = 1


      #update screen
      pygame.display.update()
      ##check if window gets closed
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
          pygame.quit()
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
        pygame.quit()




main()