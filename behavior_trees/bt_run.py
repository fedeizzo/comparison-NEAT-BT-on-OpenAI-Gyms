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

env = DerkEnv(turbo_mode=True,n_arenas=1,
   home_team=[
      { 'primaryColor': '#ff00ff', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#00ff00', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#ff0000', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],}
   ],
#    away_team=[
#       { 'primaryColor': '#ff00ff', 'slots': ['Cleavers', 'Shell', 'ParalyzingDart'],},
#       { 'primaryColor': '#00ff00', 'slots': ['Cleavers', 'Shell', 'ParalyzingDart'],},
#       { 'primaryColor': '#ff0000','slots': ['Cleavers', 'Shell', 'ParalyzingDart'],}
#    ],
   away_team=[
      { 'primaryColor': '#ff00ff', 'slots': ['Pistol', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#00ff00', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#ff0000', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],}
   ],
   
)
# create the agent with BTs 
if os.path.exists(agent_path):
    players_home = pickle.load(open(agent_path,'rb'))
else:
    pass
players_home = [BehaviorTree.generate(5) for _ in range(env.n_agents//2)]
players_away = [BehaviorTree.generate(5) for _ in range(env.n_agents//2)]
players_away[0] = BehaviorTree.from_json('C:/Users/micky/Documents/BIAI Project/bio-inspired-mutant-battlegrounds/behavior_trees/killer.JSON')
# for pl in players_home:
#     print(pl,'\n')
# for pl in players_away:
#     print(pl,'\n')
print(players_away[0])
for e in range(100):
  observation_n = env.reset()
  while True:
    action_home = [player.tick(observation_n[i])[1] for i,player in enumerate(players_home)]
    action_away = [player.tick(observation_n[i+env.n_agents//2])[1] for i,player in enumerate(players_away)]
    # action_away[0][4] = int(action_away[0][4])
    actions = action_home+action_away
    
    observation_n, reward_n, done_n, info = env.step(actions)
    if all(done_n):
        print("Episode finished")
        break
    
env.close()