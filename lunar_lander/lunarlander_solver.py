import pickle
from configparser import ConfigParser

import gymnasium as gym
import neat
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import wandb


def eval_genome(genome, config):
    env = config.env
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    runs_per_net = 5

    for _ in range(runs_per_net):
        observation, info = env.reset()

        fitness = 0
        while True:
            action = net.activate(observation)
            action = int(np.argmax(action))
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            if terminated or truncated:
                fitnesses.append(fitness)
                break
        fitnesses.append(fitness)
    # The genome's fitness is the mean performance across all runs.
    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def eval_checkpoints(checkpoints, iterations, config, gif_path):
    frames = []
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    for generation in checkpoints:
        filename = f'neat-checkpoint-{generation}'
        gif_temp_filename = f'neat-checkpoint-{generation}.gif'
        winner = neat.Checkpointer.restore_checkpoint(filename).run(eval_genomes, 1)
        winner = neat.nn.FeedForwardNetwork.create(winner, config)
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
            action = winner.activate(observation)
            action = int(np.argmax(action))
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                # stop saving frames after the first time
                frames[0].save(
                    gif_temp_filename,
                    save_all=True,
                    append_images=frames[1::4],
                    optimize=False,
                    loop=False,
                    fps=60,
                )
    for generation in checkpoints:
        gif_temp_filename = f'neat-checkpoint-{generation}.gif'
        with Image.open(gif_temp_filename) as im:
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


def eval_winner_net(winner, config):
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    fitnesses = []
    for _ in range(100):
        observation, info = env.reset()
        fitness = 0
        for _ in range(1000):
            action = winner_net.activate(observation)
            action = int(np.argmax(action))
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            if terminated or truncated:
                fitnesses.append(fitness)
                break
        fitnesses.append(fitness)
    print("Average fitness across 100 episodes is: {}".format(np.mean(fitnesses)))
    if np.mean(fitnesses) >= 200:
        print(" + The task is solved + ")
    else:
        print(" - The task is not solved - ")

    # Saves winner net
    pickle_out = open("../assets/weights/lunarlander_500_iterations.pickle", "wb")
    pickle.dump(winner, pickle_out)
    pickle_out.close()


def generate_stats(stats):
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
    return extensive_stats, species_stats


def lunar_lander_train(neat_config_path, iterations, checkpoint_frequency, use_wandb, evaluate_checkpoints):
    neat_config = ConfigParser()
    neat_config.read(neat_config_path)
    if use_wandb:
        wandb.init(
            project="Lander",
            config=neat_config.__dict__
        )
    env = gym.make("LunarLander-v2")
    observation, info = env.reset(seed=42)
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
    winner = p.run(eval_genomes, iterations)
    eval_winner_net(winner, config)
    extensive_stats, _ = generate_stats(stats)

    if use_wandb:
        for i, (c, mean, median, stdev) in enumerate(zip(stats.most_fit_genomes, stats.get_fitness_mean(),
                                                         stats.get_fitness_median(),
                                                         stats.get_fitness_stdev())):
            wandb.log({"best_fitness": c.fitness}, step=i + 1)
            wandb.log({"nodes_number": len(c.nodes)}, step=i + 1)
            wandb.log({"connections_number": len(c.connections)}, step=i + 1)
            wandb.log({"enabled_connections_number": c.size()[1]}, step=i + 1)
            wandb.log({"mean_fitness": mean}, step=i + 1)
            wandb.log({"median_fitness": median}, step=i + 1)
            wandb.log({"std_fitness": stdev}, step=i + 1)
        for i, v in enumerate(extensive_stats[['generation', 'fitness']].groupby('generation').min()):
            wandb.log({"worst_fitness": v}, step=i + 1)
        wandb.finish()
    if evaluate_checkpoints:
        eval_checkpoints([4, 9, 19, 29, 39, 49, 69, 99, 149, 199, 249, 299, 349, 399, 449, 499], iterations, config,
                         '../assets/images/neat_lunar_lander_evolution.gif')


def lunar_lander_inference(neat_config_path, winner_pickle, enable_wind, wind_power):
    env = gym.make("LunarLander-v2", render_mode="human", enable_wind=enable_wind, wind_power=wind_power)
    genome = pickle.load(open(winner_pickle, 'rb'))
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
        action = net.activate(observation)
        action = int(np.argmax(action))
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
