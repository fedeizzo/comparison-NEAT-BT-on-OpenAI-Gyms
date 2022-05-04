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
        fitness = ( fitness + 1 ) / ( total_sum + 1*len(fitnesses) )
    for id in np.random.choice(
        # np.arange(len(genomes)), size=(n_agents), replace=False
        np.arange(len(genomes)), size=(n_agents), replace=False, p=fitnesses
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

    derklings = create_derklings(genomes, config, player_class, env.n_agents, activation_functions)
    if len(derklings) != env.n_agents:
        print(len(derklings), env.n_agents)
        assert (
            len(derklings) == env.n_agents
        ), "Population for neat must be n_agents_per_arena * n_arenas"

    observation_n = env.reset()
    total_reward = []
    first_action = 0
    while True:
        if first_action < 5:
            action_n = [[1, 0, 0, 0, 0] for _ in range(env.n_agents)]
            first_action += 1
        else:
            action_n = [
                [0, 0, 0, *derklings[i].forward(observation_n[i][network_input_mask])]
                for i in range(env.n_agents)
            ]
        observation_n, reward_n, done_n, info = env.step(action_n)
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
            {"primaryColor": "#ff00ff", "slots": ["Pistol", "IronBubblegum", "HealingGland"]},
            {"primaryColor": "#00ff00", "slots": ["Pistol", "IronBubblegum", "HealingGland"]},
            # {"primaryColor": "#ff0000", "rewardFunction": {"healTeammate1": 1}},
            {"primaryColor": "#ff0000", "slots": ["Pistol", "IronBubblegum", "HealingGland"]},
        ],
        away_team=[
            {"primaryColor": "#c0c0c0", "slots": ["Pistol", "IronBubblegum", "HealingGland"]},
            {"primaryColor": "navy", "slots": ["Pistol", "IronBubblegum", "HealingGland"]},
            {"primaryColor": "red", "slots": ["Pistol", "IronBubblegum", "HealingGland"]},
        ],
    )
    # derklings = []
    # for i in range(env.n_teams):
    #     player_number = i % (len(players) - 1)
    #     type, name = (players[player_number]["path"], players[player_number]["name"])
    #     args = {k:v for k, v in players[player_number].items() if k != 'path' and k != 'name'}
    #     player = getattr(importlib.import_module(f"agent.{type}"), name)
    #     for _ in range(env.n_agents_per_team):
    #         derklings.append(
    #             player(env.n_agents_per_team, env.action_space, **args)
    #         )
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config,
    )
    config.__setattr__("env", env)
    config.__setattr__("network_input_mask", list(network_input.values()))
    config.__setattr__(
        "player_class",
        getattr(
            importlib.import_module(f"agent.neural_network_NEAT"), "DerkNeatNNPlayer"
        ),
    )
    config.__setattr__(
        "activation_functions",
        {
            0: identity,
            1: identity,
        },
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes, episodes_number)
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
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
    )
