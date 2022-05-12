import pickle
from time import time
from behavior_tree import BehaviorTree
from behavior_node import OutputIndex
from argparse import ArgumentParser
from gym_derk.envs import DerkEnv
from scipy.special import softmax
import numpy as np
import os.path
import toml
import os
from behavior_tree_evolution import *
import copy
import random
random.seed(42)

"""To evolve behavior trees we use genetic programming we use Genetic 
Algorithms principles, therefore we need:
1. genetic representation (BT)
2. population (created at the beginning of the game)
3. fitness measure (in config)
4. selection strategy (in config? or static?)
5. mutation strategy (implemented in the BT classes)
6. recombination strategy (implemented in the BT classes)
Once we have all of these, we can start the evolution process.

try with python .\behavior_trees\bt_run.py -c .\configs\bt.toml
"""


def main_dinosaurs(
    number_of_arenas,
    reward_function,
    is_train,
    episodes_number,
    bt_config,
    is_turbo,
    bt_best_player_name,
):
    # create game environment
    chrome_executable = os.environ.get("CHROMIUM_EXECUTABLE_DERK")
    chrome_executable = (
        os.expandvars(chrome_executable) if chrome_executable else None
    )
    env = DerkEnv(
        mode="normal",
        n_arenas=number_of_arenas if is_train else 6,
        reward_function=reward_function,
        turbo_mode=is_turbo,
        app_args={
            "chrome_executable": chrome_executable
            if chrome_executable
            else None
        },
        home_team=[
            {
                "primaryColor": "#ce03fc",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#8403fc",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#0331fc",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
        ],
        away_team=[
            {
                "primaryColor": "#fc1c03",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#fc6f03",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#fcad03",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
        ],
    )

    # create players at gen 0
    if is_train:
        population_size = number_of_arenas * 6
        new_population = [
            BehaviorTree.generate(5) for _ in range(population_size)
        ]
        best_player = new_population[0]
        for ep in range(episodes_number):
            start = time()
            players = new_population
            observation_n = env.reset()
            while True:
                actions = np.asarray([player.tick(observation_n[i])[1]
                                     for i, player in enumerate(players)])
                observation_n, reward_n, done_n, _ = env.step(actions)
                if all(done_n):
                    print(f"Episode {ep} finished in {time()-start}s")
                    break
            start = time()
            total_reward = env.total_reward
            for player, reward in zip(players, list(total_reward)):
                player.fitness = float(reward)
            fitnesses = [p.fitness for p in players]
            print(f"Max fitness: {max(fitnesses)}")
            players.sort(key=lambda x: x.fitness, reverse=True)
            if max(fitnesses) > best_player.fitness:
                best_player = players[0]
            # create new population
            new_population = list()

            # is elitism used?
            if bt_config["elitism"]:
                new_population += players[: bt_config["number_of_elites"]]

            print("building new population")
            # using tournament directly
            # implement crossover and mutation
            if bt_config["crossover"]:
                while len(new_population) < population_size:
                    # choose 2 parents according to their fitness
                    gen_a = tournament(players, bt_config["tournament_size"])
                    gen_b = tournament(players, bt_config["tournament_size"])
                    child = gen_b.recombination(gen_a)
                    if bt_config["mutation"]:
                        child.mutate(bt_config["mutation_rate"])
                    new_population.append(child)
            else:
                while len(new_population) > population_size:
                    gen_a = tournament(players, bt_config["tournament_size"])
                    new_individual = gen_a.copy()
                    if bt_config["mutation"]:
                        new_individual.mutate(bt_config["mutation_rate"])
                    new_population.append(new_individual)
            print(f"population mutated in {time()-start}s")

        agent_path = os.path.join(
            os.getcwd(), "behavior_trees", "saved_bts", bt_best_player_name
        )

        # save best player
        best_player.to_json(agent_path)

    else:
        assert False, "Test phase not implemented yet"

    env.close()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )
    args = p.parse_args()
    config = toml.load(args.config)

    main_dinosaurs(
        number_of_arenas=config["game"]["number_of_arenas"],
        reward_function=config["reward-function"],
        is_train=config["game"]["train"],
        episodes_number=config["game"]["episodes_number"],
        bt_config=config["bt_config"],
        is_turbo=config["game"]["fast_mode"],
        bt_best_player_name=config["game"]["bt_best_player_name"],
    )
