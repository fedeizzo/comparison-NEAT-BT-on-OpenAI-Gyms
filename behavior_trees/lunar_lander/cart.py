import os
import sys

import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavior_trees.composite_nodes import CompositeNode, composite_node_classes
from lunar_lander_condition_node import condition_node_classes
from lunar_lander_action_node import action_node_classes

name_to_class = {
    cl.__name__: cl for cl in (composite_node_classes + action_node_classes + condition_node_classes)
}
candidate = composite_node_classes + action_node_classes + condition_node_classes
from behavior_trees.lib.behavior_tree import BehaviorTree
from behavior_trees.behavior_tree_evolution import BehaviorTreeEvolution


tree = BehaviorTree.generate(5,candidate)

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
   action = tree.tick(observation)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()