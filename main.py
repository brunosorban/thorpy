# Double Q-learning algorithm with additional target network. Also uses Memory play to train the network on a bigger
# batch size instead of just one experience. Epsilon greedy algorithm is also implemented to allow the network
# to explore more actions in the beginning of the training session

import rocketDQN
from DoubleDQNAgent import Agent

env = rocketDQN.HopperEnv()

TRAIN = 0
TEST = 1
EPISODES = 50
GRAPH = True

ACTION_SPACE_SIZE = env.action_space
OBSERVATION_SPACE_SIZE = env.observation_space

FILE_TYPE = 'tf'
FILE = 'saved_networks/dqn_model281'

dqn_agent = Agent(lr=0.00075, discount_factor=0.95, num_actions=ACTION_SPACE_SIZE, epsilon=1.0, batch_size=64,
                  input_dims=OBSERVATION_SPACE_SIZE)

if TRAIN and not TEST:
    dqn_agent.train_model(env, EPISODES, GRAPH)
else:
    dqn_agent.test(env, EPISODES, FILE_TYPE, FILE, GRAPH)
