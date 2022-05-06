from argparse import ArgumentParser
import fractions
import importlib
import numpy as np
import os
from os.path import expandvars
import toml
import neat
import visualize
from scipy.special import softmax

from gym_derk.envs import DerkEnv
from activation_functions import ReLU, identity

# to save winning model
import pickle

# to flush directly in log files
import functools

print = functools.partial(print, flush=True)


def create_derklings(genomes, config, player_class, n_agents, activation_functions):
    derklings = []
    fitnesses = []
    for _, genome in genomes:
        if genome.fitness is None:
            genome.fitness = 0
        fitnesses.append(genome.fitness)
    # laplacian smoothing to avoid 0 probability
    fitnesses = softmax(fitnesses)
    total_sum = 0
    for fitness in fitnesses:
        total_sum += fitness
    for fitness in fitnesses:
        fitness = (fitness + 1) / (total_sum + 1 * len(fitnesses))
    for id in np.random.choice(
        # np.arange(len(genomes)), size=(n_agents), replace=False
        np.arange(len(genomes)),
        size=(n_agents),
        replace=False,
        p=fitnesses,
    ):
        derklings.append(
            player_class(genomes[id][1], config, activation_functions, verbose=False)
        )

    return derklings


def eval_genomes(genomes, config):
    env = config.env
    network_input_mask = config.network_input_mask
    player_class = config.player_class
    activation_functions = config.activation_functions

    derklings = []
    for _, genome in genomes:
        genome.fitness = 0
        derklings.append(
            player_class(genome, config, activation_functions, verbose=False)
        )

    if len(derklings) != env.n_agents:
        print(len(derklings), env.n_agents)
        assert (
            len(derklings) == env.n_agents
        ), "Population for neat must be n_agents_per_arena * n_arenas"

    observation_n = env.reset()
    total_reward = []
    first_action = 0
    while True:
        action_n = [
            [*derklings[i].forward(observation_n[i][network_input_mask])]
            for i in range(env.n_agents)
        ]
        observation_n, reward_n, done_n, info = env.step(action_n)
        # print(action_n[:][-2:])
        total_reward.append(np.copy(reward_n))
        if all(done_n):
            print("Episode finished")
            break

    total_reward = np.array(total_reward)
    # total_reward = total_reward.mean(axis=0) - np.sum(total_reward == 0, axis=0)
    total_reward = total_reward.sum(axis=0)
    print(total_reward)
    for (_, genome), reward in zip(genomes, list(total_reward)):
        genome.fitness = float(reward)


def main_high_level(
    players,
    number_of_arenas,
    is_turbo,
    reward_function,
    is_train,
    episodes_number,
    neat_config,
    network_input,
    weights_path,
):
    chrome_executable = os.environ.get("CHROMIUM_EXECUTABLE_DERK")
    chrome_executable = expandvars(chrome_executable) if chrome_executable else None

    env = DerkEnv(
        mode="normal",
        n_arenas=number_of_arenas,
        reward_function=reward_function,
        turbo_mode=is_turbo,
        app_args={
            "chrome_executable": chrome_executable if chrome_executable else None
        },
        home_team=[
            {
                "primaryColor": "#ff00ff",
                "slots": ["Pistol", "IronBubblegum", "HealingGland"],
            },
            {
                "primaryColor": "#00ff00",
                "slots": ["Pistol", "IronBubblegum", "HealingGland"],
            },
            # {"primaryColor": "#ff0000", "rewardFunction": {"healTeammate1": 1}},
            {
                "primaryColor": "#ff0000",
                "slots": ["Pistol", "IronBubblegum", "HealingGland"],
            },
        ],
        away_team=[
            {
                "primaryColor": "#c0c0c0",
                "slots": ["Pistol", "IronBubblegum", "HealingGland"],
            },
            {
                "primaryColor": "navy",
                "slots": ["Pistol", "IronBubblegum", "HealingGland"],
            },
            {
                "primaryColor": "red",
                "slots": ["Pistol", "IronBubblegum", "HealingGland"],
            },
        ],
    )
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config,
    )
    config.__setattr__("env", env)
    network_input_mask = list(network_input.values())
    config.__setattr__("network_input_mask", network_input_mask)
    player_class = getattr(
        importlib.import_module(f"agent.neural_network_NEAT"), "DerkNeatNNPlayer"
    )
    config.__setattr__(
        "player_class",
        player_class,
    )
    activation_functions = {
        0: identity,
        1: identity,
    }
    config.__setattr__(
        "activation_functions",
        activation_functions,
    )

    if is_train:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        # p.add_reporter(neat.Checkpointer(1, filename_prefix="neat-checkpoint-"))
        winner = p.run(eval_genomes, episodes_number)
        with open(weights_path, "wb") as f:
            pickle.dump(winner, f)
        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
    else:
        with open(weights_path, "rb") as f:
            genome = pickle.load(f)

        derklings = []
        for _ in range(env.n_agents):
            derklings.append(
                player_class(genome, config, activation_functions, verbose=False)
            )
        observation_n = env.reset()
        while True:
            action_n = [
                [*derklings[i].forward(observation_n[i][network_input_mask])]
                for i in range(env.n_agents)
            ]
            observation_n, reward_n, done_n, info = env.step(action_n)
            if all(done_n):
                print("Episode finished")
                break
    env.close()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )

    args = p.parse_args()
    config = toml.load(args.config)

    main_high_level(
        players=config["players"],
        number_of_arenas=config["game"]["number_of_arenas"],
        is_turbo=config["game"]["fast_mode"],
        reward_function=config["reward-function"],
        is_train=config["game"]["train"],
        episodes_number=config["game"]["episodes_number"],
        neat_config=config["game"]["neat_config"],
        network_input=config["network_input"],
        weights_path=config["game"]["weights_path"],
    )
