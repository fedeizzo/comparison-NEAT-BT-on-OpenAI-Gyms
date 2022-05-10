from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path
import pickle
from behavior_tree import BehaviorTree
import os

agent_path = os.path.join(os.getcwd(), "behavior_trees", "saved_bts", "killer.json")

env = DerkEnv(
    turbo_mode=True,
    n_arenas=1,
    home_team=[
        {
            "primaryColor": "#adfc03",
            "slots": ["Blaster", "FrogLegs", "HealingGland"],
        },
        {
            "primaryColor": "#3dfc03",
            "slots": ["Blaster", "FrogLegs", "HealingGland"],
        },
        {
            "primaryColor": "#03fc73",
            "slots": ["Blaster", "FrogLegs", "HealingGland"],
        },
    ],
    #    away_team=[
    #       { 'primaryColor': '#ff00ff', 'slots': ['Cleavers', 'Shell', 'ParalyzingDart'],},
    #       { 'primaryColor': '#00ff00', 'slots': ['Cleavers', 'Shell', 'ParalyzingDart'],},
    #       { 'primaryColor': '#ff0000','slots': ['Cleavers', 'Shell', 'ParalyzingDart'],}
    #    ],
    away_team=[
        {
            "primaryColor": "#fc1c03",
            "slots": ["Cleavers", "FrogLegs", "HealingGland"],
        },
        {
            "primaryColor": "#fc6f03",
            "slots": ["Blaster", "FrogLegs", "HealingGland"],
        },
        {
            "primaryColor": "#fcad03",
            "slots": ["Blaster", "FrogLegs", "HealingGland"],
        },
    ],
)
players_home = [BehaviorTree.generate(1) for _ in range(env.n_agents // 2)]
players_away = [BehaviorTree.generate(1) for _ in range(env.n_agents // 2)]
players_away[0] = BehaviorTree.from_json(agent_path)
# for pl in players_home:
#     print(pl,'\n')
# for pl in players_away:
#     print(pl,'\n')
print(players_away[0])
for e in range(100):
    observation_n = env.reset()
    while True:
        #  action_home = [player.tick(observation_n[i])[1] for i,player in enumerate(players_home)]
        #  action_away = [player.tick(observation_n[i+env.n_agents//2])[1] for i,player in enumerate(players_away)]
        #  actions = action_home+action_away
        actions = [[0, 0, 0, 0, 0] for _ in range(6)]
        actions[3] = players_away[0].tick(observation_n[3])[1]
        #  print(actions[3])
        observation_n, reward_n, done_n, info = env.step(actions)
        if all(done_n):
            print("Episode finished")
            break

env.close()
