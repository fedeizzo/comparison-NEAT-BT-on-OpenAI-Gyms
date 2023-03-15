import os
import sys

import gym
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt_lib.composite_nodes import CompositeNode, composite_node_classes
from lunar_lander.condition_nodes import condition_node_classes
from lunar_lander.action_nodes import action_node_classes
from bt_lib.behavior_tree import BehaviorTree
from bt_lib.behavior_tree_evolution import BehaviorTreeEvolution
from bt_lib.behavior_node import BehaviorStates

tree = BehaviorTree.generate(action_node_classes,condition_node_classes,composite_node_classes,1)
print(tree)
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
   state,action = tree.tick(observation)
   print(action)
   if type(action) == np.ndarray:
      action = action[0].item()
   action = int(action)
   # import pdb; pdb.set_trace()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()