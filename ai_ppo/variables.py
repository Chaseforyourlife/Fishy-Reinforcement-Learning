import torch
SPEED = 10
FPS = 30#1000 #30
FRAME_MAX =60*20

MAX_FISH_SIZE = 0 #30 #150
MIN_FISH_SIZE = -35

ALLOW_DIAGONALS= False

DEVICE = torch.device('cuda')

TRAINING_STATES = ['TRAIN_X','TRAIN_Y','TRAIN_XY','MOVE']
TRAINING_STATE = 'TRAIN_X'
X_TRAIN_Y_RANGE = 10

SHOW_GAME = False

window_size = (550,400)
MAX_FISH = 1
MAX_FISH_SPEED = 2 #6
MIN_FISH_SPEED = 1 #2
MAX_FISH_SIZE = 0 #30 #150
MIN_FISH_SIZE = -25

MAX_FISH_CONSUMED = 20

#AGENT_ALPHA = 0.0003
N=20
GAE_LAMBDA = .95
POLICY_CLIP = .2
NUM_EPOCHS=10
BATCH_SIZE = 60
MAX_MEMORY = 50_000#2_000#250#15_000#1_000_000
STARTING_MEMORY = 10_000#500#2_000#2_000
EPSILON = 1
MIN_EPSILON = 1
GAMMA = .99 #0.90 # must be less than 1

REWARD_PROXIMITY=False
REWARD_EAT=True
EPSILON_DECREASE = .0005
LEARNING_RATE = .0003#.001
END_MIN_EPSILON = 0.05

INPUT_SIZE = 4+MAX_FISH*6# 6+2+MAX_FISH*8
OUTPUT_SIZE = 5
SIZES = [INPUT_SIZE,320,160,8,OUTPUT_SIZE]


LOAD_OPTIMIZER = False

TEST = False


if TEST:
    EPSILON = .05
    MIN_EPSILON = .05
    FPS = 300
    SHOW_GAME = True