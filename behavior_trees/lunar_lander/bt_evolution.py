import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

import gymnasium as gym
from bt_lib.action_nodes import ActionNode
from bt_lib.behavior_tree import BehaviorTree
from bt_lib.composite_nodes import CompositeNode
from bt_lib.condition_nodes import ConditionNode
from tqdm import tqdm

import wandb

" cleaned up version of the code"


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
        seed=42,
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
                project="Lander",
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
        self, individual: BehaviorTree, episodes_number: int, env: gym.Env
    ):
        fitness = 0
        for _ in range(episodes_number):
            observation, info = env.reset(seed=self.seed)
            terminated = False
            while not terminated:
                state, action = individual.tick(observation)
                if action is not None:
                    action = int(action)
                else:
                    action = 0  # do nothing
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    terminated = True
                fitness += reward
        individual.fitness = (
            fitness / episodes_number
            - individual.get_size()[0] * self.tree_size_penalty
            - individual.get_size()[1] * self.tree_size_penalty
        )

    def evaluate_population(self, episodes_number: int, env: gym.Env) -> None:
        for individual in self.population:
            self.evaluate_individual(individual, episodes_number, env)
            if individual.fitness > self.best_tree.fitness:
                self.best_tree = individual

    def select_individual(self, tournament_size: int = 5) -> BehaviorTree:
        """
        Selects an individual from the population using tournament selection.
        """
        tournament = random.choices(self.population, k=tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def evolve_population(
        self, episodes_number: int, env: gym.Env
    ) -> list[BehaviorTree]:
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
            child.mutate(self.mutation_rate, True)
            # ignore non valid mutations
            if isinstance(child.root, CompositeNode):
                new_population.append(child)
        self.population = new_population

    def evalutate_folder(
        self,
        action_node_classes: list[type[ActionNode]],
        condition_node_classes: list[type[ConditionNode]],
        composite_node_classes: list[type[CompositeNode]],
        env: gym.Env,
    ):
        files = os.listdir(self.folder_path)
        files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for file in files:
            bt = BehaviorTree.from_json(
                os.path.join(self.folder_path, file),
                action_node_classes,
                condition_node_classes,
                composite_node_classes,
            )
            self.evaluate_individual(bt, self.episodes_number, env)

    def evolutionary_algorithm(self, env: gym.Env) -> BehaviorTree:
        pbar = tqdm(range(self.number_generations))
        pbar.set_description("Evolution progress")
        for i in pbar:
            self.evolve_population(
                self.episodes_number,
                env,
            )
            wandb.log({"best_fitness": self.best_tree.fitness})
            if i % self.save_every == 0:
                self.best_tree.to_json(
                    os.path.join(self.folder_path, f"best_tree_generation_{i}.json")
                )

    def __del__(self):
        if self.train:
            wandb.finish()
            self.best_tree.to_json(
                os.path.join(
                    self.folder_path, f"best_tree_{self.number_generations}.json"
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
