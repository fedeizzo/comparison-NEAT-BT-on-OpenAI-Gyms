import numpy as np

"""Class to manage evolution of the population of behavior trees.
Functionalities:

-   generate initial population
-   get evaluated population after each episode
-   return the evolved population

The evaluation is performed direclty by the Derk platform, therefore this class will take that as granted.
"""


class BehaviorTreeEvolution:
    def __init__(self, config) -> None:
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


def tournament(individuals, size):
    partecipants = list(np.random.choice(
        a=individuals, size=size, replace=False))
    partecipants.sort(key=lambda x: x.fitness, reverse=True)
    return partecipants[0]
