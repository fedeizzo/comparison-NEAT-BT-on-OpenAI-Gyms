from behavior_tree import BehaviorTree
from argparse import ArgumentParser
from gym_derk.envs import DerkEnv
from scipy.special import softmax
import numpy as np
import os.path
import toml
import os

"""To evolve behavior trees we use genetic programming we use Genetic 
Algorithms principles, therefore we need:
1. genetic representation (BT)
2. population (created at the beginning of the game)
3. fitness measure (in config)
4. selection strategy (in config? or static?)
5. mutation strategy (implemented in the BT classes)
6. recombination strategy (implemented in the BT classes)
Once we have all of these, we can start the evolution process.

try with python .\behavior_trees\bt_run.py -c .\configs\sam.toml
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
                "primaryColor": "#adfc03",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#3dfc03",
                "slots": ["Cleavers", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#03fc73",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
        ],
        away_team=[
            {
                "primaryColor": "#fc1c03",
                "slots": ["Cleavers", "FrogLegs", "HealingGland"],
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
        for _ in range(episodes_number):
            # print(new_population[0])
            players = new_population
            observation_n = env.reset()
            total_reward = []
            while True:
                actions = [
                    player.tick(observation_n[i])[1]
                    for i, player in enumerate(players)
                ]
                # print(actions)
                observation_n, reward_n, done_n, _ = env.step(actions)
                total_reward.append(np.copy(reward_n))
                if all(done_n):
                    print("Episode finished")
                    break

            total_reward = np.array(total_reward)
            total_reward = total_reward.sum(axis=0)
            # print(total_reward)
            for player, reward in zip(players, list(total_reward)):
                player.fitness = float(reward)
            players.sort(key = lambda x : x.fitness, reverse=True)
            print(players)

            # create new population
            new_population = list()

            # is elitism used?
            if bt_config["elitism"]:
                new_population += players[: bt_config["number_of_elites"]]

            fitnesses = [p.fitness for p in players]
            max_fitness = max(fitnesses)
            fitnesses = [f+max_fitness/50 for f in fitnesses]
            probabilities = softmax(fitnesses)

            print("building new population")
            # implement crossover and mutation
            if bt_config["crossover"]:
                while len(new_population) < population_size:
                    # choose 2 parents according to their fitness
                    # now implement only fitness proportional selection
                    (genitore_1, genitore_2) = np.random.choice(
                        a=players, size=2, replace=False, p=probabilities
                    )
                    genitore_1.recombination(genitore_2)
                    if bt_config["mutation"]:
                        genitore_1.mutate(bt_config["mutation_rate"])
                    new_population.append(genitore_1)
            else:
                while len(new_population) > population_size:
                    new_individual = np.random.choice(
                        a=players, size=1, replace=False, p=probabilities
                    )
                    if bt_config["mutation"]:
                        new_individual.mutate(bt_config["mutation_rate"])
                    new_population.append(new_individual)
            print("population mutated")

        agent_path = os.path.join(
            os.getcwd(), "behavior_trees", "saved_bts", bt_best_player_name
        )
        # save best player
        players[0].to_json(agent_path)

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
