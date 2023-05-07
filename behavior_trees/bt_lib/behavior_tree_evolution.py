import numpy as np
from bt_lib.behavior_tree import BehaviorTree
from bt_lib.draw import BtDrawer
from scipy.special import softmax
import random
"""Class to manage evolution of the population of behavior trees.
Functionalities:

-   generate initial population
-   get evaluated population after each episode
-   return the evolved population

The flow of the operations should be the following:
    penalize_big_trees => mu_lambda => select_parents => mutate => crossover => return new population
where the result of each phase is fed in the following phase as input. 

All of this should be performed while monitoring the performances of the various populations.
"""


class BehaviorTreeEvolution:
    def __init__(self, config:dict, pop_size:int):
        """Save all the configurations for the evolution of BTs.

        Args:
            config (dict): configuration parameters.
        """
        self.config:dict = config
        self.pop_size:int = pop_size
        self.number_of_parents:int = int(pop_size * config["parents_proportion"])
        self.previous_parents = list()
        self.hall_of_fame = list()
        self.global_best_player:BehaviorTree or None = None
        self.stats = {}
        self.iteration_nr = 0

    def generate_initial_population(self) -> list[BehaviorTree]:
        """May want to generate random population or may want to use saved
        trees.
        Size of the population is specified in the self.config.
        If test phase, generate players for one arena.

        Note: The design of initial population should be as representative as
        possible, therefore the simplest initialization possible would be to
        sample uniformly from the genotype domain.

        Note: It's useful to recognize that we can use also some previous knowledge to initialize the population, but this biases the genetic algorithm, therefore we push towards exploitation rather than exploration.
        """
        if self.config["initial_population"] == "random":
            initial_population = [
                BehaviorTree.generate(5) for _ in range(self.pop_size)
            ]
        else:
            raise NotImplementedError("non random init not implemented yet")
        return initial_population

    def evolve_population(self, population:list[BehaviorTree]) -> list[BehaviorTree]:
        """Generates the new population adopting all the policies defined in
        the configuration.

        The flow of the operations should be the following:
            penalize_big_trees => mu_lambda => select_parents => mutate => crossover => return new population
        where the result of each phase is fed in the following phase as input.

        Monitor:
        - fitness (average, maximum, variance)
        - size (average, max)
        - depth (average, max)
        - print best tree?


        Args:
            population (list[BehaviorTree]): current population.

        Returns:
            list[BehaviorTree]: next step population.
        """
        # penalize big trees
        penalized_size_population:list[BehaviorTree] = self.penalize_big_trees(population)
        # collect data
        self.iteration_nr += 1
        self.stats[f"iteration#{self.iteration_nr}"] = {}
        fitnesses, sizes, depths = list(), list(), list()
        for p in population:
            depth, count = p.get_size()
            fitnesses.append(p.fitness)
            sizes.append(count)
            depths.append(depth)
        self.stats[f"iteration#{self.iteration_nr}"]["min_fitness"] = min(fitnesses)
        self.stats[f"iteration#{self.iteration_nr}"]["max_fitness"] = max(fitnesses)
        self.stats[f"iteration#{self.iteration_nr}"]["avg_fitness"] = sum(
            fitnesses
        ) / len(fitnesses)
        self.stats[f"iteration#{self.iteration_nr}"]["max_size"] = max(sizes)
        self.stats[f"iteration#{self.iteration_nr}"]["avg_size"] = sum(sizes) / len(
            sizes
        )
        self.stats[f"iteration#{self.iteration_nr}"]["max_depth"] = max(depths)
        self.stats[f"iteration#{self.iteration_nr}"]["avg_depth"] = sum(depths) / len(
            depths
        )
        # sort population, useful later
        penalized_size_population.sort(key=lambda x: x.fitness, reverse=True)
        if self.config["monitor"]:
            print(f"iteration {self.iteration_nr}")
            print(
                f'\tFitness: min={self.stats[f"iteration#{self.iteration_nr}"]["min_fitness"]} max={self.stats[f"iteration#{self.iteration_nr}"]["max_fitness"]}  avg={self.stats[f"iteration#{self.iteration_nr}"]["avg_fitness"]}'
            )
            print(
                f'\tSize:    max={self.stats[f"iteration#{self.iteration_nr}"]["max_size"]} avg={self.stats[f"iteration#{self.iteration_nr}"]["avg_size"]}'
            )
            print(
                f'\tDepth:    max={self.stats[f"iteration#{self.iteration_nr}"]["max_depth"]} avg={self.stats[f"iteration#{self.iteration_nr}"]["avg_depth"]}'
            )
            if self.config["draw_best"]:
                drawer = BtDrawer(population[0].root)
                drawer.draw()
        # save global best player
        if (
            self.global_best_player is None
            or self.global_best_player.fitness < penalized_size_population[0].fitness
        ):
            self.global_best_player = penalized_size_population[0].copy()

        new_population:list[BehaviorTree] = list()
        # select the pool from which extract the parents
        parent_selection_pool:list[BehaviorTree] = self.mu_lambda_strategy(penalized_size_population)
        # select the parents implementing one of the thousand strategies
        # fitness proportionate, ranking, hall of fame ...
        selected_parents:list[BehaviorTree] = self.select_parents(parent_selection_pool)
        # implement elitism
        if self.config["elitism"]:
            new_population += penalized_size_population[
                : self.config["number_of_elites"]
            ]
        # grow the population with recombination and mutation
        tmp = 0
        while len(new_population) < self.pop_size:
            # here combine parents in order to get the various children for next poulation
            if self.config["crossover"]:
                # select randomly two parents from the parent pool and recombine them
                gen_a: BehaviorTree
                gen_b: BehaviorTree
                gen_a, gen_b  = np.random.choice(
                    a=selected_parents, size=2, replace=False
                )
                # no change in parents here because recombination works on deep copies
                child:BehaviorTree = gen_a.recombination(gen_b, self.config["crossover_rate"])
            else:
                # no crossover, jsut take one parent
                # since parents are less than the population size, we need to
                # iterate selected parents more than once
                child: BehaviorTree = (selected_parents[tmp]).copy()
                tmp += 1
                if tmp == len(selected_parents):
                    tmp = 0
            if self.config["mutation"]:
                # mutate the tree with the chosen strategy
                child.mutate(self.config["mutation_rate"], self.config["all_mutations"])
            new_population.append(child)
        random.shuffle(new_population)
        return new_population

    def penalize_big_trees(self, forest: list[BehaviorTree]) -> list[BehaviorTree]:
        """This function is used to penalize those trees that are too deep (or
        too big). By now we account for depth, maybe in future we'll account
        for width.

        Args:
            forest (list[BehaviorTree]): the list of trees to penalize.
        """
        for tree in forest:
            depth, count = tree.get_size()
            tree.fitness -= self.config["no_big_trees"] * (count + depth)
        return forest

    def mu_lambda_strategy(self, population: list[BehaviorTree]) -> list[BehaviorTree]:
        """Generates the next population, by taking into account the parameters
        mu and lambda. It can therefore choose how many parents to keep in the
        next population.

        It can deal also with elitism constraints.

        Args:
            population (list[BehaviorTree]): current (evaluated) population.

        Returns:
            (list[BehaviorTree]): list of parents for the next population.
        """
        selection_pool = population
        if self.config["mu_lambda_strategy"] == "plus":
            if len(self.previous_parents) != 0:
                # can be == 0 only first iteration
                selection_pool = population + self.previous_parents
        return selection_pool

    def select_parents(self, pool: list[BehaviorTree]) -> list[BehaviorTree]:
        """Selects the list of parents from the current pool.
        The pool can be composed by the current population only or by current
        plus previous parents (accordingly to the mu-lambda strategy).

        It also updates the self.previous_parents list.

        Can use:
        - random selection
        - fitness proportionate selection
        - rank based selection
        - truncated rank based selection
        - tournament
        - hall of fame

        Args:
            pool (list[BehaviorTree]): current (evaluated) parent selection
            pool.

        Returns:
            (list[BehaviorTree]): list of parents for the next population.
        """
        selected_parents = list()
        if self.config["selection_strategy"] == "random":
            selected_parents = np.random.choice(a=pool, size=(self.pop_size))
        elif self.config["selection_strategy"] == "fitness_proportionate":
            selected_parents = self.fitness_proportionate_selection(pool)
        elif self.config["selection_strategy"] == "ranking":
            selected_parents = self.ranking_selection(pool)
        elif self.config["selection_strategy"] == "truncated_ranking":
            selected_parents = self.truncated_ranking_selection(pool)
        elif self.config["selection_strategy"] == "tournament":
            selected_parents = self.tournament_selection(pool)
        elif self.config["selection_strategy"] == "hall_of_fame":
            selected_parents = self.hall_of_fame_selection(pool)
        # update previous parents
        self.previous_parents = [p.copy() for p in selected_parents]
        return selected_parents

    def fitness_proportionate_selection(self, pool : list[BehaviorTree]) -> list[BehaviorTree]:
        fitnesses = [genome.fitness for genome in pool]
        # possible negative fitnesses
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            min_fitness = -min_fitness
            fitnesses = fitnesses + min_fitness
        # # try solve problem with zeros when there's too much variance in fitnesses
        # max_fitness = max(fitnesses)
        # fitnesses = fitnesses + ((max_fitness/100)*5)
        probabilities = softmax(fitnesses)
        return np.random.choice(a=pool, size=(self.number_of_parents), p=probabilities)

    def ranking_selection(self, pool : list[BehaviorTree]) -> list[BehaviorTree]:
        pool.sort(key=lambda x: x.fitness, reverse=True)
        dividendum = sum([i + 1 for i in range(len(pool))])
        probabilities = [(1 - (i / dividendum)) for i in range(1, (len(pool) + 1))]
        from scipy.special import softmax

        probabilities = softmax(probabilities)
        return np.random.choice(a=pool, size=(self.number_of_parents), p=probabilities)

    def truncated_ranking_selection(self, pool : list[BehaviorTree]) -> list[BehaviorTree]:
        pool.sort(key=lambda x: x.fitness, reverse=True)
        return pool[: self.number_of_parents]

    def tournament_selection(self, pool : list[BehaviorTree]) -> list[BehaviorTree]:
        selected_parents = list()
        while len(selected_parents) < self.number_of_parents:
            partecipants = list(
                np.random.choice(
                    a=pool, size=self.config["tournament_size"], replace=False
                )
            )
            partecipants.sort(key=lambda x: x.fitness, reverse=True)
            selected_parents.append(partecipants[0].copy())
        return selected_parents

    def hall_of_fame_selection(self, pool : list[BehaviorTree]) -> list[BehaviorTree]:
        """Hall of fame selection strategy.
        Uses the fittest individuals in the hall of fame to be parents for the
        next population.
        If there are not enough individuals in the hall of fame, then we take
        the best from the current parent selction pool.

        Args:
            pool (list[BehaviorTree]): the current parent selection pool.

        Returns:
            list[BehaviorTree]: parents for the next population.
        """
        # update hall of fame
        pool.sort(key=lambda x: x.fitness, reverse=True)
        # add best to hall of fame
        self.hall_of_fame.append(pool.pop(0))
        # order hall of fame
        self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
        # extract the hall of fame
        selected_parents = self.hall_of_fame[: self.number_of_parents]
        # if not enough, select from current pool
        while len(selected_parents) < self.number_of_parents:
            selected_parents.append(pool.pop(0))
        return selected_parents

    def print_stats(self):
        """To be definined, print to json stats."""
        pass
