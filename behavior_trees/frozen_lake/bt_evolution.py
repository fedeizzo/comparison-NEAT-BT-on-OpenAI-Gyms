import os
import random

import gymnasium as gym
import numpy as np
from bt_lib.action_nodes import ActionNode
from bt_lib.behavior_tree import BehaviorTree
from bt_lib.composite_nodes import CompositeNode
from bt_lib.condition_nodes import ConditionNode
from PIL import Image, ImageDraw
from tqdm import tqdm

import wandb

"class that implements the genetic algorithm for evolving behavior trees"


class BehaviorTreeEvolution:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        tournament_size: int,
        elitism_rate: float,
        tree_size_penalty: float = 0.0,
        number_generations: int = 100,
        episodes_number: int = 1,
        seed=0,
        save_every=10,
        folder_path="",
        train=True,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.population: list[BehaviorTree] = []
        self.best_tree: BehaviorTree = None
        self.best_fitness_current_gen = float("-inf")
        self.mean_fitness_current_gen = 0
        self.std_fitness_current_gen = 0
        self.worst_fitness_current_gen = float("inf")
        self.tournament_size = tournament_size
        self.tree_size_penalty = tree_size_penalty
        self.number_generations = number_generations
        self.episodes_number = episodes_number
        self.seed = seed
        self.save_every = save_every
        self.folder_path = folder_path
        self.train = train
        if train:
            wandb.init(
                # set the wandb project where this run will be logged
                project="Lake",
                # track hyperparameters and run metadata
                config=self.__dict__,
            )

    def initialize_population(
        self,
        action_node_classes: list[type[ActionNode]],
        condition_node_classes: list[type[ConditionNode]],
        composite_node_classes: list[type[CompositeNode]],
        max_depth: int,
    ):
        """initialize the population with random trees"""
        self.population = []
        for _ in range(self.population_size):
            tree = BehaviorTree.generate(
                action_node_classes,
                condition_node_classes,
                composite_node_classes,
                max_depth,
            )
            self.population.append(tree)
        self.best_tree = self.population[0]

    def evaluate_individual(
        self,
        individual: BehaviorTree,
        episodes_number: int,
        env: gym.Env,
    ):
        """Evaluate the fitness of an individual using the environment"""
        fitness = 0
        for _ in range(episodes_number):
            observation, info = env.reset(seed=self.seed)
            done = False
            individual.reset()
            while not done:
                state, action = individual.tick(observation)
                if action is not None:
                    action = int(action)
                else:
                    action = 0  # do nothing
                observation, reward, terminated, truncated, info = env.step(action)
                fitness += reward
                if terminated or truncated:
                    done = True
        depth_penalty = individual.get_size()[0] * self.tree_size_penalty
        children_penalty = individual.get_size()[1] * self.tree_size_penalty
        individual.fitness = (
            fitness / episodes_number - depth_penalty - children_penalty
        )

    def save_gif(
        self,
        individual: BehaviorTree,
        env: gym.Env,
        path: str = "",
        generation: int = None,
        skip_frames: int = 4,
        fps=60,
    ):
        """play the individual in the environment and save the gif"""
        frames = []
        observation, info = env.reset(seed=self.seed)
        done = False
        individual.reset()
        while not done:
            render = env.render()
            img = Image.fromarray(render)
            if generation is not None:
                I1 = ImageDraw.Draw(img)
                I1.text(
                    (img.width // 16, img.height // 16),
                    f"Generation: {generation}",
                    fill=(0, 0, 0),
                )
            frames.append(img)
            state, action = individual.tick(observation)
            if action is not None:
                action = int(action)
            else:
                action = 0  # do nothing
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                # stop saving frames after the first time
                frames[0].save(
                    path,
                    save_all=True,
                    append_images=frames[1::skip_frames],
                    optimize=False,
                    loop=False,
                    fps=fps,
                )

    def evaluate_population(self, episodes_number: int, env: gym.Env) -> None:
        """evaluate the whole population"""
        self.best_fitness_current_gen = float("-inf")
        self.worst_fitness_current_gen = float("inf")
        self.mean_fitness_current_gen = 0
        for individual in self.population:
            self.evaluate_individual(individual, episodes_number, env)
            if individual.fitness > self.best_tree.fitness:
                self.best_tree = individual
            if individual.fitness > self.best_fitness_current_gen:
                self.best_fitness_current_gen = individual.fitness
            if individual.fitness < self.worst_fitness_current_gen:
                self.worst_fitness_current_gen = individual.fitness
            self.mean_fitness_current_gen += individual.fitness
        self.mean_fitness_current_gen /= self.population_size
        self.std_fitness_current_gen = np.std([x.fitness for x in self.population])

    def select_individual(self, tournament_size: int = 5) -> BehaviorTree:
        """
        Selects an individual from the population using tournament selection.
        """
        tournament = random.choices(self.population, k=tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def evolve_population(
        self, episodes_number: int, env: gym.Env
    ) -> list[BehaviorTree]:
        """Applies simple ES to evolve the population"""
        self.evaluate_population(episodes_number, env)
        new_population = []
        sorted_pop_fitness = sorted(
            self.population, key=lambda x: x.fitness, reverse=True
        )
        new_population.extend(
            sorted_pop_fitness[: int(self.population_size * self.elitism_rate)]
        )
        while len(new_population) < self.population_size:
            parent1: BehaviorTree = self.select_individual()
            parent2: BehaviorTree = self.select_individual()
            child: BehaviorTree = parent1.recombination(parent2, self.crossover_rate)
            child.mutate(self.mutation_rate)
            # ignore invalid mutations
            if isinstance(child.root, CompositeNode):
                new_population.append(child)

        self.population = new_population

    def evalutate_folder(
        self,
        action_node_classes: list[type[ActionNode]],
        condition_node_classes: list[type[ConditionNode]],
        composite_node_classes: list[type[CompositeNode]],
        env: gym.Env,
        generations_to_save: list[int] = [],
        path: str = "",
        skip_frames: int = 4,
        fps=60,
    ):
        files = os.listdir(self.folder_path)
        files = [file for file in files if file.endswith(".json")]
        files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for file in files:
            generation = int(file.split("_")[-1].split(".")[0])
            if generation in generations_to_save:
                bt = BehaviorTree.from_json(
                    os.path.join(self.folder_path, file),
                    action_node_classes,
                    condition_node_classes,
                    composite_node_classes,
                )
                self.save_gif(
                    bt,
                    env,
                    path=os.path.join(path, str(generation) + ".gif"),
                    generation=generation,
                    skip_frames=skip_frames,
                    fps=fps,
                )
        self.concat_gifs(path, generations_to_save, fps)

    def concat_gifs(self, path, generations_to_save, fps: int = 60):
        # extract all the frames from the gif
        frames = []
        for generation in generations_to_save:
            with Image.open(os.path.join(path, f"{generation}.gif")) as im:
                try:
                    while 1:
                        im.seek(im.tell() + 1)
                        frames.append(im.copy())
                except EOFError:
                    pass  # end of sequence
        # save the frames as a new gif
        frames[0].save(
            os.path.join(path, "evolution.gif"),
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            loop=0,
            fps=10,
        )

    def evolutionary_algorithm(self, env: gym.Env) -> BehaviorTree:
        pbar = tqdm(range(self.number_generations))
        pbar.set_description("Evolution progress")
        for i in pbar:
            self.evolve_population(
                self.episodes_number,
                env,
            )

            wandb.log({"best_fitness overall": self.best_tree.fitness}, step=i)
            wandb.log({"best_fitness": self.best_fitness_current_gen}, step=i)
            wandb.log({"mean_fitness": self.mean_fitness_current_gen}, step=i)
            wandb.log({"std_fitness": self.std_fitness_current_gen}, step=i)
            wandb.log({"worst_fitness": self.worst_fitness_current_gen}, step=i)
            wandb.log({"best_tree_depth": self.best_tree.get_size()[0]}, step=i)
            wandb.log({"best_tree_children": self.best_tree.get_size()[1]}, step=i)
            wandb.log(
                {"best_tree_executed_nodes": self.best_tree.get_executed_nodes()},
                step=i,
            )
            if i % self.save_every == 0:
                self.best_tree.to_json(
                    os.path.join(self.folder_path, f"best_tree_generation_{i}.json")
                )

    def __del__(self):
        if self.train:
            wandb.finish()
            self.best_tree.to_json(
                os.path.join(
                    self.folder_path,
                    f"best_tree_generation_{self.number_generations}.json",
                )
            )


if __name__ == "__main__":
    bt = BehaviorTreeEvolution(10, 0.1, 0.7, 5, 0.2)
    from bt_lib.action_nodes import ActionNode
    from bt_lib.composite_nodes import CompositeNode, composite_node_classes
    from bt_lib.condition_nodes import ConditionNode
    from lunar_lander.action_nodes import action_node_classes
    from lunar_lander.condition_nodes import condition_node_classes

    bt.initialize_population(
        action_node_classes, condition_node_classes, composite_node_classes, 5
    )
    env = gym.make("LunarLander-v2")
    bt.evaluate_population(10, env)
    print(bt.best_tree.fitness)
    print(bt.best_tree)
    bt.evolve_population(10, env)
