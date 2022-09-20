import torch
SPEED = 10
FPS = 600 #30
FRAME_MAX = 60

MAX_FISH_SIZE = 0 #30 #150
MIN_FISH_SIZE = -35

ALLOW_DIAGONALS= False

DEVICE = torch.device('cuda')

TRAINING_STATES = ['TRAIN_X','TRAIN_Y','TRAIN_XY','MOVE']
TRAINING_STATE = 'TRAIN_XY'
X_TRAIN_Y_RANGE = 10

SHOW_GAME = False

window_size = (550,400)
MAX_FISH = 1
MAX_FISH_SPEED = 2 #6
MIN_FISH_SPEED = 1 #2
MAX_FISH_SIZE = 0 #30 #150
MIN_FISH_SIZE = -35

MAX_MEMORY = 150_000#250#15_000#1_000_000
STARTING_MEMORY = 10_000#2_000#2_000
EPSILON = 1
MIN_EPSILON = 1
GAMMA = .9 #0.90 # must be less than 1

REWARD_PROXIMITY=True

BATCH_SIZE = 60
LEARNING_RATE = .001#.001
END_MIN_EPSILON = 0.05

INPUT_SIZE = 4+MAX_FISH*6# 6+2+MAX_FISH*8
HIDDEN_SIZE = 64
HIDDEN2_SIZE = 32

LOAD_OPTIMIZER = True

TEST = False


if TEST:
    EPSILON = .05
    MIN_EPSILON = .05
    FPS = 15
    SHOW_GAME = True