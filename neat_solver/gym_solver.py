import pickle
from configparser import ConfigParser
from typing import Optional

import gymnasium as gym
import neat
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import wandb


def create_gym_env(
    env_name: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    **env_kwargs,
) -> gym.Env:
    """Creates a Gym environment given its name and configuration.

    Args:
        env_name (str): The name of the Gym environment to create.
        seed (int): The seed to use for the environment.
        render_mode (Optional[str]): The render mode to use for the environment. Defaults to None.
        **env_kwargs: Additional keyword arguments to pass to the environment.

    Returns:
        gym.Env: The Gym environment.
    """
    env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
    if seed is not None:
        _, _ = env.reset(seed=seed)
    return env


def eval_genome(
    genome: neat.DefaultGenome, config: neat.Config, runs_per_net: int = 5
) -> float:
    """Evaluates a genome by running it in the environment a set number of times.

    Args:
        genome (neat.DefaultGenome): The genome to evaluate.
        config (neat.Config): The NEAT config to build the network from.
        runs_per_net (int, optional): The number of times to run the genome in the environment. Defaults to 5.

    Returns:
        float: The mean fitness across all runs.
    """
    env = config.env
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for _ in range(runs_per_net):
        observation, info = env.reset()

        fitness = 0
        while True:
            if isinstance(observation, int):
                observation = [observation]
            action = net.activate(observation)
            action = int(np.argmax(action))
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            if terminated or truncated:
                break
        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes: list[neat.DefaultGenome], config: neat.Config):
    """Evaluates a list of genomes by running them in the environment."""
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def eval_checkpoints(
    env_name: str,
    checkpoints: list[int],
    config: neat.Config,
    gif_path: str,
    **env_kwargs,
):
    """Evaluates a list of checkpoints by running them in the environment and saving the results as a gif.

    Args:
        env_name (str): The name of the Gym environment to create.
        checkpoints (list[int]): A list of checkpoint generations to evaluate.
        config (neat.Config): The NEAT config to build the network from.
        gif_path (str): The path to save the gif to.
        **env_kwargs: Additional keyword arguments to pass to the environment.
    """
    env = create_gym_env(env_name, render_mode="rgb_array", **env_kwargs)

    frames = []
    for generation in checkpoints:
        filename = f"neat-checkpoint-{generation}"
        winner_genome = neat.Checkpointer.restore_checkpoint(filename).run(
            eval_genomes, 1
        )
        winner = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        observation, info = env.reset(seed=1)
        done = False
        while not done:
            render = env.render()
            img = Image.fromarray(render)
            if generation is not None:
                I1 = ImageDraw.Draw(img)
                I1.text(
                    (img.width // 16, img.height // 16),
                    f"Generation: {generation}",
                    fill=(255, 255, 255),
                )
            frames.append(img)
            if isinstance(observation, int):
                observation = [observation]
            action = winner.activate(observation)
            action = int(np.argmax(action))
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                # stop saving frames after the first time
                frames[0].save(
                    f"{filename}.gif",
                    save_all=True,
                    append_images=frames[1::4],
                    optimize=False,
                    loop=False,
                    fps=60,
                )
    for generation in checkpoints:
        with Image.open(f"neat-checkpoint-{generation}.gif") as im:
            try:
                while 1:
                    im.seek(im.tell() + 1)
                    frames.append(im.copy())
            except EOFError:
                pass  # end of sequence
    # save the frames as a new gif
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        loop=0,
        fps=60,
    )


def eval_winner_net(
    winner: neat.DefaultGenome,
    config: neat.Config,
    env: gym.Env,
    winner_dump_path: str,
    num_evals: int = 100,
):
    """Evaluates a winner net by running it in the environment a set number of times.

    Args:
        winner (neat.DefaultGenome): The winner genome to evaluate.
        config (neat.Config): The NEAT config to build the network from.
        env (gym.Env): The environment to run the winner net in.
        winner_dump_path (str): The path to save the winner genome to.
        num_evals (int, optional): The number of times to run the genome in the environment. Defaults to 100.
    """
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    fitnesses = []
    for _ in range(num_evals):
        observation, info = env.reset()
        fitness = 0
        while True:
            if isinstance(observation, int):
                observation = [observation]
            action = winner_net.activate(observation)
            action = int(np.argmax(action))
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            if terminated or truncated:
                fitnesses.append(fitness)
                break
        fitnesses.append(fitness)

    print(f"Average fitness across {num_evals} episodes is: {np.mean(fitnesses)}")

    with open(winner_dump_path, "wb") as outfile:
        pickle.dump(winner, outfile)


def generate_stats(stats):
    """Generates extensive and species stats from the NEAT stats object."""
    extensive_stats = []
    # stats.generation_statistics has len() == number of iterations
    for gen, i in enumerate(stats.generation_statistics):
        for species_id, members in i.items():
            for net_id, fitness in members.items():
                extensive_stats.append([gen, species_id, net_id, fitness])

    extensive_stats = pd.DataFrame(
        extensive_stats, columns=["generation", "species", "genome", "fitness"]
    )

    # species stats
    species_stats = [
        [gen, curve] for gen, curve in enumerate(stats.get_species_sizes())
    ]
    species_stats = pd.DataFrame(species_stats, columns=["generation", "species_sizes"])
    return extensive_stats, species_stats


def gym_train(
    env_name: str,
    env_kwargs: dict,
    neat_config_path: str,
    num_iterations: int,
    checkpoint_frequency: int,
    use_wandb: bool,
    evaluate_checkpoints: bool,
    winner_dump_path: str,
    gif_path: str,
    gif_checkpoints: list[int],
):
    """Trains a lunar lander agent using NEAT.

    Args:
        env_name (str): Name of the Gym environment to use.
        env_kwargs (str): Additional keyword arguments to pass to the environment.
        neat_config_path (str): The path to the NEAT config file.
        num_iterations (int): The number of iterations to train for.
        checkpoint_frequency (int): The frequency to save checkpoints.
        use_wandb (bool): Whether to use Weights and Biases to log the training.
        evaluate_checkpoints (bool): Whether to create an output gif image.
    """
    neat_config = ConfigParser()
    neat_config.read(neat_config_path)

    if use_wandb:
        wandb.init(project=env_name, config=neat_config.__dict__)

    env = create_gym_env(env_name, **env_kwargs)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    config.__setattr__("env", env)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    if evaluate_checkpoints:
        p.add_reporter(neat.Checkpointer(checkpoint_frequency))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, num_iterations)
    eval_winner_net(winner, config, env, winner_dump_path)
    extensive_stats, _ = generate_stats(stats)

    if use_wandb:
        for i, (c, mean, median, stdev) in enumerate(
            zip(
                stats.most_fit_genomes,
                stats.get_fitness_mean(),
                stats.get_fitness_median(),
                stats.get_fitness_stdev(),
            )
        ):
            wandb.log({"best_fitness": c.fitness}, step=i + 1)
            wandb.log({"nodes_number": len(c.nodes)}, step=i + 1)
            wandb.log({"connections_number": len(c.connections)}, step=i + 1)
            wandb.log({"enabled_connections_number": c.size()[1]}, step=i + 1)
            wandb.log({"mean_fitness": mean}, step=i + 1)
            wandb.log({"median_fitness": median}, step=i + 1)
            wandb.log({"std_fitness": stdev}, step=i + 1)

        for i, v in enumerate(
            extensive_stats[["generation", "fitness"]].groupby("generation").min()
        ):
            wandb.log({"worst_fitness": v}, step=i + 1)

        wandb.finish()

    if evaluate_checkpoints:
        eval_checkpoints(env_name, gif_checkpoints, config, gif_path, **env_kwargs)


def gym_inference(
    env_name: str,
    env_kwargs: dict,
    neat_config_path: str,
    winner_pickle: str,
):
    """Runs the lunar lander agent in inference mode rendered for humans.

    Args:
        env_name (str): Name of the Gym environment to use.
        env_kwargs (str): Additional keyword arguments to pass to the environment.
        neat_config_path (str): The path to the NEAT config file.
        winner_pickle (str): The path to the winner genome pickle file.
    """
    env = create_gym_env(
        env_name,
        render_mode="human",
        **env_kwargs,
    )
    genome = pickle.load(open(winner_pickle, "rb"))
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    observation, info = env.reset()

    while True:
        if isinstance(observation, int):
            observation = [observation]
        action = net.activate(observation)
        action = int(np.argmax(action))
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
