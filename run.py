from argparse import ArgumentParser
import importlib
import numpy as np
import os
from os.path import expandvars
import toml
import neat
import visualize
from scipy.special import softmax

import asyncio
from gym_derk import DerkSession, DerkAgentServer, DerkAppInstance
from gym_derk.envs import DerkEnv


async def run_player(env: DerkSession, DerkPlayerClass, class_args):
    """
    Runs a DerkPlayer
    """
    player = DerkPlayerClass(env.n_agents, env.action_space, class_args)
    obs = await env.reset()
    player.signal_env_reset(obs)
    ordi = await env.step()

    while not env.done:
        actions = player.take_action(ordi)
        ordi = await env.step(actions)


async def main_low_level(p1, p2, n, turbo, reward_function):
    """
    Runs the game in n arenas between p1 and p2
    """
    chrome_executable = os.environ.get("CHROMIUM_EXECUTABLE_DERK")
    chrome_executable = expandvars(chrome_executable) if chrome_executable else None

    p1_type, p1_name = (p1["path"], p1["name"])
    p2_type, p2_name = (p1["path"], p1["name"])
    player1 = getattr(importlib.import_module(f"agent.{p1_type}"), p1_name)
    player2 = getattr(importlib.import_module(f"agent.{p2_type}"), p2_name)
    del p1["name"]
    del p2["name"]
    del p1["path"]
    del p2["path"]

    agent_p1 = DerkAgentServer(
        run_player, args={"DerkPlayerClass": player1, "class_args": p1}, port=8788
    )
    agent_p2 = DerkAgentServer(
        run_player, args={"DerkPlayerClass": player2, "class_args": p2}, port=8789
    )

    await agent_p1.start()
    await agent_p2.start()

    app = DerkAppInstance(
        chrome_executable=chrome_executable if chrome_executable else None
    )
    await app.start()

    await app.run_session(
        n_arenas=n,
        turbo_mode=turbo,
        agent_hosts=[
            {"uri": agent_p1.uri, "regions": [{"sides": "home"}]},
            {"uri": agent_p2.uri, "regions": [{"sides": "away"}]},
        ],
        reward_function=reward_function,
    )
    await app.print_team_stats()


def eval_genomes(genomes, config):
    env = config.env
    network_input_mask = config.network_input_mask
    derklings = []
    fitnesses = []
    for _, genome in genomes:
        if genome.fitness is None:
            genome.fitness = 0
        fitnesses.append(genome.fitness)
    fitnesses = softmax(fitnesses)
    for id in np.random.choice(
        np.arange(len(genomes)), size=(env.n_agents), replace=True, p=fitnesses
    ):
        derklings.append(neat.nn.FeedForwardNetwork.create(genomes[id][1], config))
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
            pred = [
                derklings[i].activate(observation_n[i][network_input_mask])
                for i in range(env.n_agents)
            ]
            casts = [pred[i][0:2] for i in range(env.n_agents)]
            casts_i = [np.argmax(cast) for cast in casts]
            focuses = [pred[i][2:10] for i in range(env.n_agents)]
            focuses_i = [np.argmax(focus) for focus in focuses]
            action_n = [
                [0, 0, 0, cast_i if cast[cast_i] >0 else 0, focus_i if focus[focus_i] >0 else 0 ] for i,cast,cast_i,focus,focus_i in zip(pred,casts,casts_i,focuses,focuses_i)
            ]
            # action_n += [
            #     [0, 0, 0, 1, 0] for _ in range(3)
            # ]
        print(len(action_n))
        observation_n, reward_n, done_n, info = env.step(action_n)
        print(observation_n[0][network_input_mask])
        print(action_n[0])
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
            {"primaryColor": "#ff00ff", "slots": ["Pistol", None, None]},
            {"primaryColor": "#00ff00", "slots": ["Pistol", None, None]},
            # {"primaryColor": "#ff0000", "rewardFunction": {"healTeammate1": 1}},
            {"primaryColor": "#ff0000", "slots": ["Pistol", None, None]},
        ],
        away_team=[
            {"primaryColor": "#c0c0c0", "slots": ["Pistol", None, None]},
            {"primaryColor": "navy", "slots": ["Pistol", None, None]},
            {"primaryColor": "red", "slots": ["Pistol", None, None]},
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

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes, episodes_number)
    # visualize.draw_net(config, winner, True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)
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
