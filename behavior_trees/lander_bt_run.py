from argparse import ArgumentParser

import gymnasium as gym
import toml
from bt_lib.composite_nodes import composite_node_classes
from lunar_lander.action_nodes import action_node_classes
from lunar_lander.bt_evolution import BehaviorTreeEvolution
from lunar_lander.condition_nodes import condition_node_classes


def main_lander(lander_config):
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
        train=lander_config["game"]["train"],
    )

    if lander_config["game"]["train"]:
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
