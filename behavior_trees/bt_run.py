# test BT to see if it runs

from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path
import pickle
from behavior_tree import BehaviorTree

agent_path = './bt_agent.pkl'

env = DerkEnv(turbo_mode=True,n_arenas=1)
# create the agent with BTs 
if os.path.exists(agent_path):
    players_home = pickle.load(open(agent_path,'rb'))
else:
    pass
players_home = [BehaviorTree.generate(5) for _ in range(env.n_agents//2)]
players_away = [BehaviorTree.generate(5) for _ in range(env.n_agents//2)]
for e in range(100):
  observation_n = env.reset()
  while True:
    action_home = [player.tick(observation_n[i])[1] for i,player in enumerate(players_home)]
    action_away = [player.tick(observation_n[i+env.n_agents//2])[1] for i,player in enumerate(players_away)]
    actions = action_home+action_away
    observation_n, reward_n, done_n, info = env.step(actions)
    if all(done_n):
        print("Episode finished")
        break
    
env.close()