# supposedly agents that are controlled by neural network
# the network evolves randomly

# Game is here for Docs http://docs.gym.derkgame.com/

# the games is played in "episodes" or "epochs"
# For each episode the agents are fixed, but if they "learn" through RL or other mechanisms, they can improve from the environment. 
# Each episode is split in steps, where each step is a "tick". 
# Each agent is given a fixed set of actions, and the environment is updated accordingly when the agents perfom these actions. 
# This repeats for each tick of an episode.
# Each agent can learn from the results of each tick, try new actions etc.
# After each episode ends, the agents are "reset" and the environment is also reset. The agents may "evolve" in some way.

# import of classic libraries for the game

from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path

# creation of the environment in Chromium (side note, it is required to keep the tab open, or the game will not update)),
# 3 agents each of the 2 teams, 2 totems ( default game, default rules, default map, default settings etc)
# agents are controlled by neural network
# each agent is generated with random equipment ( arms , misc, tail). Each slot may be filled or not, giving each agent different "abilities". 
# Our network should be able to correctly exploit any of these abilities and come up with a "good" strategy to destroy the enemy and win.

# Win is defined by whichever team has the most points when times is up. 
# Standard reward function promotes killing and destroying totems. All other actions have 0 reward 

# Colors and aesthetics are present but randomly initalized and are not meaningful for the outcome. However they are useful to distinguish the agents.
env = DerkEnv(turbo_mode=True)


# standard NN from Derk Docs
class Network:
  def __init__(self, weights=None, biases=None):
    self.network_outputs = 13
    if weights is None:
      weights_shape = (self.network_outputs, len(ObservationKeys))
      self.weights = np.random.normal(size=weights_shape)
    else:
      self.weights = weights
    if biases is None:
      self.biases = np.random.normal(size=(self.network_outputs))
    else:
      self.biases = biases

  def clone(self):
    return Network(np.copy(self.weights), np.copy(self.biases))

  def forward(self, observations):
    outputs = np.add(np.matmul(self.weights, observations), self.biases)
    casts = outputs[3:6]
    cast_i = np.argmax(casts)
    focuses = outputs[6:13]
    focus_i = np.argmax(focuses)
    return (
      math.tanh(outputs[0]), # MoveX
      math.tanh(outputs[1]), # Rotate
      max(min(outputs[2], 1), 0), # ChaseFocus
      (cast_i + 1) if casts[cast_i] > 0 else 0, # CastSlot
      (focus_i + 1) if focuses[focus_i] > 0 else 0, # Focus
    )

  def copy_and_mutate(self, network, mr=0.1):
    self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
    self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)


# loads weights, we need to load our weights from the file and save them
weights_path = 'Project\bio-inspired-mutant-battlegrounds\agent\neural network\weights.npy'
biases_path = 'Project\bio-inspired-mutant-battlegrounds\agent\neural network\biases.npy'
weights = np.load('weights_path') if os.path.isfile('weights_path') else None
biases = np.load('biases_path') if os.path.isfile('biases_path') else None

# each Derk Agent is controlled by a neural network!
networks = [Network(weights, biases) for i in range(env.n_agents)]

# set the mode 
env.mode = 'train' # test

# set the number of episodes
episodes = 10
for e in range(episodes):
  # reset the environment at each episode
  observation_n = env.reset()
  while True:
    # get the actions from the networks
    action_n = [networks[i].forward(observation_n[i]) for i in range(env.n_agents)]
    # update
    observation_n, reward_n, done_n, info = env.step(action_n)
    
    if all(done_n):
        print("Episode finished")
        break
    
  # this is the part that needs to be changed with neat
  if env.mode == 'train':
    reward_n = env.total_reward
    print(reward_n)
    top_network_i = np.argmax(reward_n)
    top_network = networks[top_network_i].clone()
    for network in networks:
      network.copy_and_mutate(top_network)
    print('top reward', reward_n[top_network_i])
    
    # save network
    np.save('weights_path', top_network.weights)
    np.save('biases_path', top_network.biases)
env.close()