from argparse import ArgumentParser

import toml
from derk.derk_solver import derk_main_high_level
from lunar_lander.lunarlander_solver import lunar_lander_inference, lunar_lander_train

ENVIRONMENTS = ["derk", "lunarlander"]


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-e",
        "--environment",
        help="Derk environment to use",
        type=str,
        required=True,
        choices=ENVIRONMENTS,
    )
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )
    p.add_argument("-i", "--inference", help="Inference mode", action="store_true")

    args = p.parse_args()
    if args.environment == "derk":
        config = toml.load(args.config)
        derk_main_high_level(
            players=config["players"],
            number_of_arenas=config["game"]["number_of_arenas"],
            is_turbo=config["game"]["fast_mode"],
            reward_function=config["reward-function"],
            is_train=config["game"]["train"],
            episodes_number=config["game"]["episodes_number"],
            neat_config=config["game"]["neat_config"],
            network_input=config["network_input"],
            best_stats_path=config["game"]["best_stats"],
            extensive_stats_path=config["game"]["extensive_stats"],
            species_stats_path=config["game"]["species_stats"],
            weights_path=config["game"]["weights_path"],
        )
    elif args.environment == "lunarlander":
        config = toml.load(args.config)
        if args.inference:
            lunar_lander_inference(
                config["game"]["neat_config"],
                config["game"]["winner_pickle"],
                config["game"]["enable_wind"],
                config["game"]["wind_power"],
            )
        else:
            lunar_lander_train(
                config["game"]["neat_config"],
                config["game"]["iterations"],
                config["game"]["checkpoint_frequency"],
                config["game"]["use_wandb"],
                config["game"]["evaluate_checkpoints"],
            )
