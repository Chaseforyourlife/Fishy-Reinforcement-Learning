from pickle import TRUE
import torch
SPEED = 5
FPS = 0#60#1000 #30



ALLOW_DIAGONALS=False

RANDOM_START=False
TRAINING_STATES = ['TRAIN_X','TRAIN_Y','TRAIN_XY','MOVE']
TRAINING_STATE = 'Move'
X_TRAIN_Y_RANGE = 10
XY_TRAIN_RANGE = 0
SHOW_GAME = True
if TRAINING_STATE == 'TRAIN_XY':
    RANDOM_START =True


window_size = (550,400)
MAX_FISH = 1
MAX_FISH_SPEED = 6 #6
MIN_FISH_SPEED = 2 #2
MAX_FISH_SIZE = 5#150 #30 #150
MIN_FISH_SIZE = -25

MAX_FISH_CONSUMED = MAX_FISH_SIZE+5
FRAME_MAX =250*MAX_FISH_CONSUMED+1500*MAX_FISH_SIZE#0#240

#AGENT_ALPHA = 0.0003
N=20
GAE_LAMBDA = .95
POLICY_CLIP = .2
NUM_EPOCHS=1 # try 10 and 5?
BATCH_SIZE = 60


GAMMA = .95 #0.90 # must be less than 1, originally .99

REWARD_PROXIMITY=False
REWARD_EAT=True

LEARNING_RATE = .0003#.01#.0003#.0003#.0003#.001


INPUT_SIZE = 4+MAX_FISH*6# 6+2+MAX_FISH*8
if ALLOW_DIAGONALS:
    OUTPUT_SIZE = 9
else:
    OUTPUT_SIZE = 5
SIZES = [INPUT_SIZE,1000,1000,OUTPUT_SIZE]


LOAD_MODEL = True

TEST =False
if TEST:
    FPS = 600
    SHOW_GAME = True
    LOAD_MODEL=True