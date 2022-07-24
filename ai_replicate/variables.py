import pygame

window_size=(550,400)

SPEED = 10
FPS = 1800
MAX_FISH = 1
MAX_FISH_SIZE = 0 #30 #150
MIN_FISH_SIZE = -35
TIME_MAX=20/SPEED

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