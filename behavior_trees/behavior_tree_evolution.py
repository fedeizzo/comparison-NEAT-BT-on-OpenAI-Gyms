import numpy as np
from behavior_tree import BehaviorTree

"""Class to manage evolution of the population of behavior trees.
Functionalities:

-   generate initial population
-   get evaluated population after each episode
-   return the evolved population

The flow of the operations should be the following:
    penalize_big_trees => mu_lambda => select_parents => mutate => crossover => return new population
where the result of each phase is fed in the following phase as input. 
"""


class BehaviorTreeEvolution:
    def __init__(self, config):
        """Save all the configurations for the evolution of BTs.

        Args:
            config (dict): configuration parameters.
        """
        self.config = config

    def generate_initial_population(self):
        """May want to generate random population or may want to use saved
        trees.
        Size of the population is specified in the self.config.
        If test phase, generate players for one arena.

        Note: The design of initial population should be as representative as
        possible, therefore the simplest initialization possible would be to
        sample uniformly from the genotype domain.

        Note: It's useful to recognize that we can use also some previous knowledge to initialize the population, but this biases the genetic algorithm, therefore we push towards exploitation rather than exploration.
        """
        pass

    def penalize_big_trees(self, forest: "list[BehaviorTree]"):
        """This function is used to penalize those trees that are too deep (or
        too big). By now we account for depth, maybe in future we'll account 
        for width.

        Args:
            forest (list[BehaviorTree]): the list of trees to penalize.
        """
        for tree in forest:
            depth, count = tree.get_size()
            tree.fitness -= (self.config["no_big_trees"]*depth)
        return forest

    def mutate(self,  population: "list[BehaviorTree]"):
        """Mutate the population if requested.

        Args:
            population (list[BehaviorTree]): the population to be mutated.

        Returns:
            (list[BehaviorTree]): the mutated population
        """
        return population

    def crossover(self,  population: "list[BehaviorTree]"):
        """Performs crossover on the elements of the population if requested.

        Args:
            population (list[BehaviorTree]): the population to be crossed over.

        Returns:
            (list[BehaviorTree]): the crossed-over population.
        """
        return population

    def mu_lambda_strategy(self,  population: "list[BehaviorTree]"):
        """Generates the next population, by taking into account the parameters
        mu and lambda. It can therefore choose how many parents to keep in the 
        next population.

        It can deal also with elitism constraints.

        Args:
            population (list[BehaviorTree]): current (evaluated) population.

        Returns:
            (list[BehaviorTree]): list of parents for the next population.
        """
        return population

    def select_parents(self, population: "list[BehaviorTree]"):
        """Selects the list of parents from the current population.
        Can use:
        - random selection
        - fitness proportionate selection
        - rank based selection
        - truncated rank based selection
        - tournament
        - hall of fame

        Args:
            population (list[BehaviorTree]): current (evaluated) population.

        Returns:
            (list[BehaviorTree]): list of parents for the next population.
        """
        return population


def tournament(individuals, size):
    partecipants = list(np.random.choice(
        a=individuals, size=size, replace=False))
    partecipants.sort(key=lambda x: x.fitness, reverse=True)
    return partecipants[0]
