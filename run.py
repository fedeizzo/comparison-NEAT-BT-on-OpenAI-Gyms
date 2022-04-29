from argparse import ArgumentParser
import importlib
import os
from os.path import expandvars
import toml

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


def main_high_level(
    players, number_of_arenas, is_turbo, reward_function, is_train, episodes_number
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
            {"primaryColor": "#ff00ff"},
            {"primaryColor": "#00ff00", "slots": ["Talons", None, None]},
            {"primaryColor": "#ff0000", "rewardFunction": {"healTeammate1": 1}},
        ],
        away_team=[
            {"primaryColor": "#c0c0c0"},
            {"primaryColor": "navy", "slots": ["Talons", None, None]},
            {"primaryColor": "red", "rewardFunction": {"healTeammate1": 1}},
        ],
    )
    derklings = []
    for i in range(env.n_teams):
        type, name = (players[i]["path"], players[i]["name"])
        del players[i]["path"]
        del players[i]["name"]
        player = getattr(importlib.import_module(f"agent.{type}"), name)
        for _ in env.n_agents_per_team:
            derklings.append(player(players[i]))

    for e in range(episodes_number):
        observation_n = env.reset()
        while True:
            action_n = [
                derklings[i].take_actions(observation_n[i]) for i in range(env.n_agents)
            ]
            observation_n, reward_n, done_n, info = env.step(action_n)
            if all(done_n):
                print("Episode finished")
                break

        if is_train:
            # EVOLVE DERKLINGS
            # SAVE SOME WEIGHTS
            # OTHER STUFF
            pass
    env.close()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )

    args = p.parse_args()
    config = toml.load(args.config)

    main_high_level(
        config["players"],
        config["game"]["number_of_arenas"],
        config["game"]["fast_mode"],
        config["reward-function"],
        config["game"]["train"],
        config["game"]["episodes_number"],
    )
