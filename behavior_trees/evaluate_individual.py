import gymnasium as gym
import numpy as np
import toml
from argparse import ArgumentParser
from bt_lib.behavior_node import BehaviorStates
from bt_lib.behavior_tree import BehaviorTree
from tqdm import tqdm

# from bt_lib.behavior_tree_evolution import BehaviorTreeEvolution
from bt_lib.composite_nodes import CompositeNode, composite_node_classes
from lunar_lander.action_nodes import action_node_classes
from lunar_lander.condition_nodes import condition_node_classes
from lunar_lander.bt_evolution import BehaviorTreeEvolution
import random

def main_lander(lander_config):
    random.seed(lander_config["bt_config"]["mutation_seed"])
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    bt_evolution = BehaviorTreeEvolution(
        lander_config["bt_config"]["population_size"],
        lander_config["bt_config"]["mutation_rate"],
        lander_config["bt_config"]["crossover_rate"],
        lander_config["bt_config"]["tournament_size"],
        lander_config["bt_config"]["elitism_rate"],
        lander_config["bt_config"]["tree_size_penalty"],
        lander_config["bt_config"]["number_generations"],
        lander_config["bt_config"]["episodes_number"],
        lander_config["bt_config"]["seed"],
        lander_config["bt_config"]["save_every"],
        lander_config["game"]["folder_path"],
        False,
    )
    bt = BehaviorTree.from_json("lander/best_tree_generation_1000.json",action_node_classes,condition_node_classes,
                                composite_node_classes)


    bt_evolution.save_gif(bt,env,path="eval.gif",generation=1000)
    print(bt.fitness)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )
    args = p.parse_args()
    config = toml.load(args.config)
    main_lander(config)
