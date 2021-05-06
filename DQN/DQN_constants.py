import numpy as np
epsilon = 1 #randomness
START_EPSILON = epsilon #for statistics
EPSILONE_DECAY = 0.999999 #every episode will be epsilon*EPISODE_DECAY
min_epsilon = 0.05
LEARNING_RATE = 0.5
DISCOUNT = 0.99
CONST_LAST_FRAME_REWARD = -np.Inf

# MODEL_NAME = 'red_blue_16X32X64X9'
MODEL_NAME = 'red_blue_16X32X64X9_2'

