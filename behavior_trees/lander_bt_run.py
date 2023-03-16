import gym
import numpy as np
from bt_lib.behavior_node import BehaviorStates
from bt_lib.behavior_tree import BehaviorTree
from bt_lib.behavior_tree_evolution import BehaviorTreeEvolution
from bt_lib.composite_nodes import CompositeNode, composite_node_classes
from lunar_lander.action_nodes import action_node_classes
from lunar_lander.condition_nodes import condition_node_classes

if __name__ == "__main__":
    tree = BehaviorTree.generate(
        action_node_classes, condition_node_classes, composite_node_classes, 1
    )
    print(tree)
    env = gym.make("LunarLander-v2")
    observation, info = env.reset(seed=42, return_info=True)

    for _ in range(1000):
        state, action = tree.tick(observation)
        print(action)

        if action is not None:
            action = int(action)
        else:
            action = 0  # do nothing

        env.render()
        observation, reward, terminated, info = env.step(action)

        if terminated:
            observation, info = env.reset(return_info=True)
    env.close()
