import os
import random
from argparse import ArgumentParser

import gymnasium as gym
import toml
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
        save_every=lander_config["bt_config"]["save_every"],
        folder_path=lander_config["game"]["folder_path"],
        train=not inference,
        prob_keep_not_executed=lander_config["bt_config"]["prob_keep_not_executed"],
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
        files = sorted(
            [
                int(i.split(".")[0].split("_")[-1])
                for i in os.listdir(lander_config["game"]["folder_path"])
                if os.path.isfile(os.path.join(lander_config["game"]["folder_path"], i))
            ]
        )
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        bt_evolution.evalutate_folder(
            action_node_classes,
            condition_node_classes,
            composite_node_classes,
            env,
            files,
            os.path.join(lander_config["game"]["folder_path"], "results_gif"),
        )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )
    p.add_argument("-i", "--inference", help="Inference mode", action="store_true")
    args = p.parse_args()
    config = toml.load(args.config)
    main_lander(config, args.inference)
