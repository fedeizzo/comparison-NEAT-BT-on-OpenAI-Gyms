from argparse import ArgumentParser
import importlib
import os
from os.path import expandvars
import toml

import asyncio
from gym_derk import DerkSession, DerkAgentServer, DerkAppInstance


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


async def main(p1, p2, n, turbo, reward_function):
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


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-c", "--config", help="Path to config file", type=str, required=True
    )

    args = p.parse_args()
    config = toml.load(args.config)

    asyncio.get_event_loop().run_until_complete(
        main(
            config["players"][0],
            config["players"][1],
            config["game"]["number_of_arenas"],
            config["game"]["fast_mode"],
            config["reward-function"],
        )
    )
