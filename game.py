import pygame


#VARIABLES#
window_size = (550,400)
fishy_left_image = pygame.image.load('static/images/fishy_left.png')
fishy_right_image = pygame.image.load('static/images/fishy_right.png')
screen = pygame.display.set_mode(window_size)

class Fishy:
  def __init__(self):
    self.width = 30
    self.height = 8 
    self.x = 225
    self.y = 200
    self.direction:char = 'R'
    self.x_speed = 0
    self.y_speed = 0
    self.max_x_speed = 8
    self.max_y_speed = 8
    self.x_speed_change = .25
    self.y_speed_change = .25
    self.image_left = pygame.transform.scale(fishy_left_image,(self.width,self.height)) 
    self.image_right = pygame.transform.scale(fishy_right_image,(self.width,self.height)) 


  def move(self):
    print(self.x_speed)
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
      self.direction = 'L'
    if keys[pygame.K_RIGHT]:
      self.x_speed += .25
      self.direction = 'R'  
    if keys[pygame.K_UP]:
        self.y_speed -= .25
    if keys[pygame.K_DOWN]:
      self.y_speed += .25
    #if speed is below speed_change, set speed to 0
    if abs(self.x_speed) < self.x_speed_change:
      self.x_speed = 0 
    if abs(self.y_speed) < self.y_speed_change:
      self.y_speed = 0 
  def draw(self,screen):
    screen.blit(self.image_left if self.direction == 'L' else self.image_right,(self.x,self.y))



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
  small_fish_image = pygame.image.load('static/images/fish.png')
  #screen.fill((255,255,255))
  
  #pygame.draw.rect(screen,(0,0,0),(225,200,40,20))
  clock = pygame.time.Clock()

  main_fish = Fishy()



  running = True
  while running:
    
    screen.blit(fishy_background,(0,0))




    clock.tick(30)
    main_fish.move()
    main_fish.draw(screen)

    pygame.display.update()
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
        pygame.quit()




main()