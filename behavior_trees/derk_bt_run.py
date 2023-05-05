import os
import os.path
from argparse import ArgumentParser
from time import time

import numpy as np
import toml
from bt_lib.behavior_tree import BehaviorTree
from bt_lib.behavior_tree_evolution import BehaviorTreeEvolution
from bt_lib.composite_nodes import composite_node_classes
from derk.action_nodes import action_node_classes
from derk.condition_nodes import condition_node_classes
from gym_derk.envs import DerkEnv

import wandb

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
    episodes_number,
    bt_config,
    is_turbo,
    start_bt_config,
    best_player,
    use_wandb,
    inference,
):
    # create game environment
    chrome_executable = os.environ.get("CHROMIUM_EXECUTABLE_DERK")
    chrome_executable = os.expandvars(chrome_executable) if chrome_executable else None
    env = DerkEnv(
        mode="normal",
        n_arenas=number_of_arenas if not inference else 1,
        reward_function=reward_function,
        turbo_mode=is_turbo,
        app_args={
            "chrome_executable": chrome_executable if chrome_executable else None
        },
        home_team=[
            {
                "primaryColor": "#ce03fc",
                "slots": ["Pistol", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#8403fc",
                "slots": ["Pistol", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#0331fc",
                "slots": ["Pistol", "FrogLegs", "HealingGland"],
            },
        ],
        away_team=[
            {
                "primaryColor": "#fc1c03",
                "slots": ["Pistol", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#fc6f03",
                "slots": ["Pistol", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#fcad03",
                "slots": ["Pistol", "FrogLegs", "HealingGland"],
            },
        ],
    )

    if use_wandb:
        wandb.init(project="Derk", config=config)
    if not inference:
        population_size = number_of_arenas * 6
    else:
        population_size = 6
    evolution_engine = BehaviorTreeEvolution(bt_config, population_size)
    # create players at gen 0
    if not inference:
        new_population = [
            # BehaviorTree.generate(5) for _ in range(population_size)
            BehaviorTree.from_json(
                start_bt_config,
                action_node_classes,
                condition_node_classes,
                composite_node_classes,
            )
            for _ in range(population_size)
        ]

        for ep in range(episodes_number):
            start = time()
            players = new_population

            # for index, player in enumerate(players):
            #     player.to_json(f"./behavior_trees/saved_bts/dummy_gen_{ep}_player_{index}.json")

            observation_n = env.reset()
            while True:
                # print(f"+ Episode {ep} -- New Step")

                actions = []
                for i, player in enumerate(players):
                    # print(f"- Player {i}")
                    actions.append(player.tick(observation_n[i])[1])

                actions = np.asarray(actions)

                observation_n, reward_n, done_n, _ = env.step(actions)

                if all(done_n):
                    print(f"Episode {ep} finished in {int(time()-start)}s")
                    break
            start = time()
            total_reward = env.total_reward
            max_fitness = float("-inf")
            best_player = None
            for player, reward in zip(players, list(total_reward)):
                player.fitness = float(reward)
                if reward > max_fitness:
                    max_fitness = reward
                    best_player = player

            if use_wandb:
                wandb.log({"mean_fitness": np.mean(total_reward, axis=0)}, step=ep)
                wandb.log({"max_fitness": np.max(total_reward, axis=0)}, step=ep)
                wandb.log({"min_fitness": np.min(total_reward, axis=0)}, step=ep)
                wandb.log({"std_fitness": np.std(total_reward, axis=0)}, step=ep)
                if best_player:
                    depth, size = best_player.get_size()
                    wandb.log({"best_tree_depth": depth}, step=ep)
                    wandb.log({"best_tree_size": size}, step=ep)

            new_population = evolution_engine.evolve_population(players)
            print(f"population mutated in {int(time()-start)}s")


        # save best player
        evolution_engine.global_best_player.to_json(best_player)

    else:
        # load best player
        new_population = [
            # BehaviorTree.generate(5) for _ in range(population_size)
            BehaviorTree.from_json(
                best_player,
                action_node_classes,
                condition_node_classes,
                composite_node_classes,
            )
            for _ in range(population_size)
        ]
        observation_n = env.reset()
        while True:
            actions = []
            for i, player in enumerate(new_population):
                actions.append(player.tick(observation_n[i])[1])
                # import pdb; pdb.set_trace()
            actions = np.asarray(actions)
            # import pdb; pdb.set_trace()
            observation_n, reward_n, done_n, _ = env.step(actions)

            if all(done_n):
                print(f"Episode finished")
                break

    env.close()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )
    p.add_argument("-i", "--inference", help="Inference mode", action="store_true")
    args = p.parse_args()
    config = toml.load(args.config)

    main_dinosaurs(
        number_of_arenas=config["game"]["number_of_arenas"],
        reward_function=config["reward-function"],
        episodes_number=config["game"]["episodes_number"],
        bt_config=config["bt_config"],
        is_turbo=config["game"]["fast_mode"],
        start_bt_config=config["game"]["starting_config"],
        best_player=config["game"]["best_player"],
        use_wandb=config["game"]["use_wandb"],
        inference=args.inference,
    )
