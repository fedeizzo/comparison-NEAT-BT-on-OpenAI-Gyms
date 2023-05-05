import os
import random
from argparse import ArgumentParser

import gymnasium as gym
import toml
from bt_lib.behavior_tree import BehaviorTree
from bt_lib.composite_nodes import composite_node_classes
from lunar_lander.action_nodes import action_node_classes
from lunar_lander.bt_evolution import BehaviorTreeEvolution
from lunar_lander.condition_nodes import condition_node_classes


def main_lander(lander_config: dict, inference: bool):
    random.seed(lander_config["bt_config"]["mutation_seed"])
    bt_evolution = BehaviorTreeEvolution(
        population_size=lander_config["bt_config"]["population_size"],
        mutation_rate=lander_config["bt_config"]["mutation_rate"],
        crossover_rate=lander_config["bt_config"]["crossover_rate"],
        tournament_size=lander_config["bt_config"]["tournament_size"],
        elitism_rate=lander_config["bt_config"]["elitism_rate"],
        tree_size_penalty=lander_config["bt_config"]["tree_size_penalty"],
        number_generations=lander_config["bt_config"]["number_generations"],
        episodes_number=lander_config["bt_config"]["episodes_number"],
        seed=lander_config["bt_config"]["seed"],
        best_player=lander_config["game"]["best_player"],
        train=not inference,
        prob_keep_not_executed=lander_config["bt_config"]["prob_keep_not_executed"],
        use_wandb=lander_config["game"]["use_wandb"],
    )

    if not inference:
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
        player_path = config["game"]["best_player"]
        bt = BehaviorTree.from_json(
            player_path,
            action_node_classes,
            condition_node_classes,
            composite_node_classes,
        )
        env = gym.make("LunarLander-v2", render_mode="human")
        bt_evolution.evaluate_individual(bt, 1, env)
        # env_1 = gym.make("LunarLander-v2", render_mode="rgb_array")
        # bt_evolution.save_gif(bt, env_1, "lake.gif", skip_frames=1, fps=30)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )
    p.add_argument("-i", "--inference", help="Inference mode", action="store_true")
    args = p.parse_args()
    config = toml.load(args.config)
    main_lander(config, args.inference)
