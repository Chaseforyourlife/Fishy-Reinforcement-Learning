import torch

SPEED = 5
FPS = 0#1000 #30
MAX_GAME_LIMIT=0
SAVE_MODEL = True


ALLOW_DIAGONALS=True

RANDOM_START=False
TRAINING_STATES = ['TRAIN_X','TRAIN_Y','TRAIN_XY','MOVE']
TRAINING_STATE = 'MOVE'
X_TRAIN_Y_RANGE = 10
XY_TRAIN_RANGE = 0
SHOW_GAME = True
if TRAINING_STATE == 'TRAIN_XY':
    RANDOM_START =True



DEATH_ON_CONTACT = True
SHIFT_LAST_ADVANTAGE = False
ADD_LAST_STATE = True
SHOW_STATE_SCREEN = False

window_size = (550,400)
window_resize = (200,275)
MAX_FISH = 4
MAX_FISH_SPEED = 6 #6
MIN_FISH_SPEED = 2 #2
#MAX_FISH_SIZE = 0#150 #30 #150
START_MAX_FISH_SIZE = 0#60#150
INCREMENT_FISH_SIZE = True#True#False
NUM_GAMES_INCREMENT_START = 250
MIN_FISH_SIZE = -25

GAMES_BEFORE_SAVE = 0
#MAX_FISH_CONSUMED = MAX_FISH_SIZE+5


    

FRAME_MAX =0#0#60#250*MAX_FISH_CONSUMED+1500*MAX_FISH_SIZE#0#240

#AGENT_ALPHA = 0.0003
N=20
GAE_LAMBDA = .95
POLICY_CLIP = .2
NUM_EPOCHS=1 # try 10 and 5?
BATCH_SIZE = 16#10000#15 is fastest
TIMESTEPS_PER_ITERATION=500

GAMMA = .975 #0.90 # must be less than 1, originally .99

REWARD_PROXIMITY=False
REWARD_EAT=True

LEARNING_RATE = .00006#.0006#.01#.0003#.0003#.0003#.001





RELATIVE_STATES = False

if RELATIVE_STATES:
    INPUT_SIZE = 4+MAX_FISH*6# 6+2+MAX_FISH*8
else:
    INPUT_SIZE = 4+MAX_FISH*8# 6+2+MAX_FISH*8

if ALLOW_DIAGONALS:
    OUTPUT_SIZE = 9
else:
    OUTPUT_SIZE = 5
#SIZES = [INPUT_SIZE,2000,2000,2000,OUTPUT_SIZE]
SIZES=[INPUT_SIZE,400,200,100,50,OUTPUT_SIZE]

PREV_FRAME_NUMBER = 1

FRAME_SKIP = False

LOAD_MODEL = True

#default optimizer is Adam

###OPTUNA VARIBALES
OPTUNA = False
OPTUNA_MIN_MAX = 'maximize' # 'minimize' or 'maximize'
OPTUNA_NUM_TRIALS = 10

#switch to test for list?
OPTUNA_MODEL =  [[3,6],[1000,3000]]
OPTUNA_LR =  False # [.00001,.001] #example used .0003
OPTUNA_GAMMA = False # [.90,.99]
OPTUNA_BATCH_SIZE = False # [10,1000]
OPTUNA_OPTIM = False # ["Adam","RMSprop","SGD"]
OPTUNA_EPOCHS = False # [1,30]
OPTUNA_DROPOUT = False # [.2,.5]

if OPTUNA:
    SAVE_MODEL=False
    MAX_GAME_LIMIT = 1000
    LOAD_MODEL=False


TEST =False
if TEST:
    FPS = 120
    SHOW_GAME = True
    LOAD_MODEL=True
    OPTUNA = False
    SAVE_MODEL=False
