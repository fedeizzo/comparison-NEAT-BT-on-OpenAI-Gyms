from argparse import ArgumentParser
from enum import Enum

import toml
from derk.derk_solver import derk_main_high_level
from gym_solver import gym_inference, gym_train


# https://stackoverflow.com/questions/43968006/support-for-enum-arguments-in-argparse
class EnvType(Enum):
    DERK = "derk"
    LUNARLANDER = "lunarlander"
    FROZENLAKE = "frozenlake"

    def __str__(self):
        return self.value


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-e",
        "--environment",
        help="Gym environment to use",
        type=EnvType,
        choices=list(EnvType),
        required=True,
    )
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )
    p.add_argument("-i", "--inference", help="Inference mode", action="store_true")

    args = p.parse_args()
    if args.environment == EnvType.DERK:
        config = toml.load(args.config)
        derk_main_high_level(
            players=config["players"],
            number_of_arenas=config["game"]["number_of_arenas"],
            is_turbo=config["game"]["fast_mode"],
            reward_function=config["reward-function"],
            is_train=not args.inference,
            episodes_number=config["game"]["episodes_number"],
            neat_config=config["game"]["neat_config"],
            network_input=config["network_input"],
            best_stats_path=config["game"]["best_stats"],
            extensive_stats_path=config["game"]["extensive_stats"],
            species_stats_path=config["game"]["species_stats"],
            weights_path=config["game"]["weights_path"],
        )
    elif args.environment in [EnvType.LUNARLANDER, EnvType.FROZENLAKE]:
        config = toml.load(args.config)
        if args.inference:
            gym_inference(
                config["env"]["name"],
                config["env"]["kwargs"],
                config["game"]["neat_config"],
                config["game"]["winner_pickle"],
            )
        else:
            gym_train(
                config["env"]["name"],
                config["env"]["kwargs"],
                config["game"]["neat_config"],
                config["game"]["num_iterations"],
                config["game"]["checkpoint_frequency"],
                config["game"]["use_wandb"],
                config["game"]["evaluate_checkpoints"],
                config["game"]["winner_pickle"],
                config["game"]["gif_path"],
                config["game"]["gif_checkpoints"],
            )
