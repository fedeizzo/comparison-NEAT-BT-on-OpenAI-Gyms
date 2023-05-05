import importlib
import os
from pathlib import Path
import pickle
from os.path import expandvars

import neat
import numpy as np
import pandas as pd
import visualize
from derk.generate_all_possible_actions import generate_actions
from gym_derk.envs import DerkEnv


def eval_genomes(genomes, config):
    env = config.env
    network_input_mask = config.network_input_mask
    player_class = config.player_classes[0]
    player_args = {
        k.replace("p0_", ""): v
        for k, v in config.__dict__.items()
        if k.startswith("p0_")
    }

    derklings = []
    for _, genome in genomes:
        genome.fitness = 0
        derklings.append(player_class(genome, config, **player_args))

    if len(derklings) != env.n_agents:
        print(len(derklings), env.n_agents)
        assert (
            len(derklings) == env.n_agents
        ), "Population for neat must be n_agents_per_arena * n_arenas"

    observation_n = env.reset()
    total_reward = []
    while True:
        action_n = [
            [*derklings[i].forward(observation_n[i][network_input_mask])]
            for i in range(env.n_agents)
        ]
        observation_n, reward_n, done_n, _ = env.step(action_n)
        total_reward.append(np.copy(reward_n))
        if all(done_n):
            print("Episode finished")
            break

    total_reward = np.array(total_reward)
    total_reward = total_reward.sum(axis=0)
    print(total_reward)
    for (_, genome), reward in zip(genomes, list(total_reward)):
        genome.fitness = float(reward)


def set_player_args(player_id, p_cfg, config):
    if p_cfg["name"] == "DerkQLearningNEATPlayer":
        actions_space = p_cfg["actions_space"]
        movement_split = p_cfg["movement_split"]
        rotation_split = p_cfg["rotation_split"]
        chase_focus_split = p_cfg["chase_focus_split"]
        del p_cfg["actions_space"]
        del p_cfg["movement_split"]
        del p_cfg["rotation_split"]
        del p_cfg["chase_focus_split"]
        p_cfg["all_actions"] = generate_actions(
            actions_space, movement_split, rotation_split, chase_focus_split
        )
    for k, v in p_cfg.items():
        if k != "name" and k != "path":
            config.__setattr__(f"p{player_id}_{k}", v)


def derk_main_high_level(
    players,
    number_of_arenas,
    is_turbo,
    reward_function,
    is_train,
    episodes_number,
    neat_config,
    network_input,
    best_stats_path,
    extensive_stats_path,
    species_stats_path,
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
                "slots": ["Magnum", "IronBubblegum", "HealingGland"],
            },
            # {"primaryColor": "#ff0000", "rewardFunction": {"healTeammate1": 1}},
            {
                "primaryColor": "#ff0000",
                "slots": ["Blaster", "IronBubblegum", "HealingGland"],
            },
        ],
        away_team=[
            {
                "primaryColor": "#c0c0c0",
                "slots": ["Pistol", "IronBubblegum", "HealingGland"],
            },
            {
                "primaryColor": "navy",
                "slots": ["Magnum", "IronBubblegum", "HealingGland"],
            },
            {
                "primaryColor": "red",
                "slots": ["Blaster", "IronBubblegum", "HealingGland"],
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
    config.pop_size = env.n_agents
    config.__setattr__("env", env)
    network_input_mask = list(network_input.values())
    config.__setattr__("network_input_mask", network_input_mask)
    player_classes = [
        getattr(importlib.import_module(f"derk.agent.{p['path']}"), p["name"])
        for p in players
    ]
    config.__setattr__(
        "player_classes",
        player_classes,
    )
    for i, p in enumerate(players):
        set_player_args(i, p, config)

    if is_train:
        assert all(
            [p["name"] for p in players]
        ), "During train all players must be the same"
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        # p.add_reporter(neat.Checkpointer(1, filename_prefix="neat-checkpoint-"))
        winner = p.run(eval_genomes, episodes_number)
        os.makedirs(Path(weights_path).parent, exist_ok=True)
        with open(weights_path, "wb") as f:
            pickle.dump(winner, f)
        # visualize.draw_net(config, winner, False)
        best_genome_stats = pd.DataFrame()
        # stats_df['species'] = stats.get_species_sizes()
        # stats_df['species_fitness'] = stats.get_species_fitness()
        # BEST GENOME STATS
        most_fit_genomes = [c for c in stats.most_fit_genomes]
        best_genome_stats["generation"] = np.arange(len(most_fit_genomes))
        best_genome_stats["fitness"] = [g.fitness for g in most_fit_genomes]
        best_genome_stats["nodes_number"] = [len(g.nodes) for g in most_fit_genomes]
        best_genome_stats["connections_number"] = [
            len(g.connections) for g in most_fit_genomes
        ]
        best_genome_stats["enabled_connections_number"] = [
            g.size()[1] for g in most_fit_genomes
        ]
        best_genome_stats["mean"] = stats.get_fitness_mean()
        best_genome_stats["median"] = stats.get_fitness_median()
        best_genome_stats["stdev"] = stats.get_fitness_stdev()

        # ALL GENOMES STATS
        extensive_stats = []
        # stats.generation_statistics has len() == number of iterations
        for gen, i in enumerate(stats.generation_statistics):
            for species_id, members in i.items():
                for net_id, fitness in members.items():
                    extensive_stats.append([gen, species_id, net_id, fitness])

        extensive_stats = pd.DataFrame(
            extensive_stats, columns=["generation", "species", "genome", "fitness"]
        )

        # SPECIES STATS
        species_stats = [
            [gen, curve] for gen, curve in enumerate(stats.get_species_sizes())
        ]
        species_stats = pd.DataFrame(
            species_stats, columns=["generation", "species_sizes"]
        )

        best_genome_stats.to_pickle(best_stats_path)
        extensive_stats.to_pickle(extensive_stats_path)
        species_stats.to_pickle(species_stats_path)
    else:
        with open(weights_path, "rb") as f:
            genome = pickle.load(f)

        derklings = []
        for i in range(env.n_teams):
            del players[i]["name"]
            del players[i]["path"]
            for _ in range(env.n_agents_per_team):
                derklings.append(player_classes[i](genome, config, **players[i]))
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
