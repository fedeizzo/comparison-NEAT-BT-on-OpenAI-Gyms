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
        self.population = []
        self.fitness = []
        self.best_tree:BehaviorTree = None
        self.best_fitness = -np.inf

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
    @staticmethod
    def evaluate_individual(individual: BehaviorTree, episodes_number : int, env: gym.Env) -> list[float]:
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
        return fitness / episodes_number
    
    def evaluate_population(self, episodes_number : int, env: gym.Env) -> None:
        self.fitness = []
        for individual in self.population:
            fitness = BehaviorTreeEvolution.evaluate_individual(individual, episodes_number, env)
            self.fitness.append(fitness)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_tree = individual
                
    def select_individual(self,tournament_size : int = 5) -> BehaviorTree:
        '''
        Selects an individual from the population using tournament selection.
        '''
        tournament = random.choices(list(zip(self.population,self.fitness)), k=tournament_size)
        return max(tournament, key=lambda x: x[1])[0]
        
    def evolve_population(self, episodes_number : int, env: gym.Env) -> list[BehaviorTree]:
        self.evaluate_population(episodes_number, env)
        new_population = []
        for _ in range(int(self.population_size * self.elitism_rate)):
            best_individual = self.population[np.argmax(self.fitness)]
            new_population.append(best_individual)
        for _ in range(int(self.population_size * (1 - self.elitism_rate))):
            parent1:BehaviorTree = self.select_individual()
            parent2:BehaviorTree = self.select_individual()
            child:BehaviorTree = parent1.recombination(parent2, self.crossover_rate)
            child.mutate(self.mutation_rate, True)
            new_population.append(child)
        self.population = new_population