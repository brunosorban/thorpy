# Double Q-learning algorithm with additional target network. Also uses Memory play to train the network on a bigger
# batch size instead of just one experience. Epsilon greedy algorithm is also implemented to allow the network
# to explore more actions in the beginning of the training session

import rocket
# import numpy as np
# import matplotlib.pyplot as plt
from agent import Agent


# # Test of the ODE and step() + reset() fcn
# duration = 20
# N = 200
#
# env = rocket.HopperEnv()
#
# state = env.reset()
# print(f"The initial state of the hopper is: {state}")
#
# S = []
# t = np.linspace(0, duration, N)
#
# for i in enumerate(t):
#     new_state, reward, done = env.step(0.298, state)
#     state = new_state
#     S.append(new_state[0])
#
# print(S[199])
# plt.plot(t, S)
# plt.show()

env = rocket.HopperEnv()

TRAIN = 1
TEST = 0
EPISODES = 500
GRAPH = True

ACTION_SPACE_SIZE = env.action_space
OBSERVATION_SPACE_SIZE = env.observation_space

FILE_TYPE = 'tf'
FILE = 'saved_networks/dqn_model23'

dqn_agent = Agent(lr=0.0075, discount_factor=0.99, num_actions=ACTION_SPACE_SIZE, epsilon=1.0, batch_size=128,
                  input_dims=OBSERVATION_SPACE_SIZE)

if TRAIN and not TEST:
    dqn_agent.train_model(env, EPISODES, GRAPH)
else:
    dqn_agent.test(env, EPISODES, FILE_TYPE, FILE, GRAPH)
