import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bt_lib.behavior_tree import BehaviorTree
from bt_lib.composite_nodes import CompositeNode
from bt_lib.action_nodes import ActionNode
from bt_lib.condition_nodes import ConditionNode
import gymnasium as gym
import numpy as np
import random

" cleaned up version of the code"


class BehaviorTreeEvolution:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        tournament_size: int,
        elitism_rate: float,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.population:list[BehaviorTree] = []
        self.best_tree:BehaviorTree = None
        self.tournament_size = tournament_size
        self.best_tree = None

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
    @staticmethod
    def evaluate_individual(individual: BehaviorTree, episodes_number : int, env: gym.Env):
        fitness = 0
        for _ in range(episodes_number):
            observation, info = env.reset()
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
        individual.fitness = fitness/episodes_number
    
    def evaluate_population(self, episodes_number : int, env: gym.Env) -> None:
        for individual in self.population:
            BehaviorTreeEvolution.evaluate_individual(individual, episodes_number, env)
            if individual.fitness > self.best_tree.fitness:
                self.best_tree = individual

    def select_individual(self,tournament_size : int = 5) -> BehaviorTree:
        '''
        Selects an individual from the population using tournament selection.
        '''
        tournament = random.choices(self.population, k=tournament_size)
        return max(tournament, key=lambda x: x.fitness)
        
    def evolve_population(self, episodes_number : int, env: gym.Env) -> list[BehaviorTree]:
        self.evaluate_population(episodes_number, env)
        new_population = []
        sorted_pop_fitness = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population.extend(sorted_pop_fitness[:int(self.population_size * self.elitism_rate)])
        while len(new_population) < self.population_size:
            parent1:BehaviorTree = self.select_individual()
            parent2:BehaviorTree = self.select_individual()
            # print(parent1.fitness, parent2.fitness)
            # print(parent1, parent2)
            child:BehaviorTree = parent1.recombination(parent2, self.crossover_rate)
            # print(child)
            child.mutate(self.mutation_rate, False)
            # print(child)
            # ignore non valid mutations
            if isinstance(child.root, CompositeNode):
                new_population.append(child)
        self.population = new_population
        
if __name__ == "__main__":
    
    bt = BehaviorTreeEvolution(10, 0.1, 0.7, 5, 0.2)
    from bt_lib.action_nodes import ActionNode
    from bt_lib.condition_nodes import ConditionNode
    from bt_lib.composite_nodes import CompositeNode
    from lunar_lander.action_nodes import action_node_classes
    from lunar_lander.condition_nodes import condition_node_classes
    from bt_lib.composite_nodes import composite_node_classes
    bt.initialize_population(action_node_classes, condition_node_classes, composite_node_classes, 5)
    env = gym.make("LunarLander-v2")
    bt.evaluate_population(10, env)
    print(bt.best_tree.fitness)
    print(bt.best_tree)
    bt.evolve_population(10, env)