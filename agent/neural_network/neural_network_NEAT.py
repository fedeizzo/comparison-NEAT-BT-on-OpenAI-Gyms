from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path

from neural_network.pytorch_neat.recurrent_net import RecurrentNet

env = DerkEnv(turbo_mode=True)


# loads weights, we need to load our weights from the file and save them
weights_path = 'Project\bio-inspired-mutant-battlegrounds\agent\neural network\weights.npy'
biases_path = 'Project\bio-inspired-mutant-battlegrounds\agent\neural network\biases.npy'
weights = np.load('weights_path') if os.path.isfile('weights_path') else None
biases = np.load('biases_path') if os.path.isfile('biases_path') else None

# each Derk Agent is controlled by a neural network!
networks = [RecurrentNet(196,0,13) for i in range(env.n_agents)]

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
