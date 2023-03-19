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
from bt_lib.bt_evolution import BehaviorTreeEvolution


def main_lander(lander_config):
    if lander_config["game"]["train"]:
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
        )

        env = gym.make("LunarLander-v2")

        bt_evolution.initialize_population(
            action_node_classes,
            condition_node_classes,
            composite_node_classes,
            lander_config["bt_config"]["max_depth"],
        )
        # create progress bar
        bt_evolution.evolutionary_algorithm(env)
        env.close()
    else:
        env = gym.make("LunarLander-v2", render_mode="human")
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
        bt_evolution.evalutate_folder(
            action_node_classes, condition_node_classes, composite_node_classes, env
        )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )
    args = p.parse_args()
    config = toml.load(args.config)
    main_lander(config)
