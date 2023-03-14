from behavior_tree import BehaviorTree
from behavior_node import OutputIndex
from argparse import ArgumentParser
from gym_derk.envs import DerkEnv
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
        n_arenas=number_of_arenas if is_train else 1,
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
        
        try:
            population = [BehaviorTree.from_json(bt_config['game']['starting_config']) for _ in range(population_size)]
        except (KeyError, FileNotFoundError):
            population = [BehaviorTree.generate(5) for _ in range(population_size)]
            
        population_home = population[:population_size//2]
        population_away = population[population_size//2:]
        
        for i in range(episodes_number):
            observation_n = env.reset()
            while True:
                actions_home = np.asarray([player.tick(observation_n[i])[1]for i, player in enumerate(population_home)])
                actions_away = np.asarray([player.tick(observation_n[i])[1]for i, player in enumerate(population_away)])
                actions = np.asarray([*actions_home,*actions_away])
                observation_n, reward_n, done_n, _ = env.step(actions)
                if all(done_n):
                    print(f"Episode {i} finished")
                    break
            total_reward = env.total_reward
            for player, reward in zip(population, list(total_reward)):
                player.fitness = float(reward)
            # create new population and evolve
        agent_path = os.path.join(
            os.getcwd(), "behavior_trees", "saved_bts", bt_best_player_name
        )
        players = [*population_home, *population_away]
        players.sort(key=lambda x: x.fitness, reverse=True)
        # save best player
        print(players[0])
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
