# Behavior Trees Evolution

Here we report the results of the experiments and the discussions deveoped during the impementation of the BT architecture.

## Overview

We developed and braind new BT platform for BTs specialized for this very task. This platform offers the following features:

1. classes and methods for the basic structures of BT
    - action nodes
    - composite nodes
    - condition nodes (not yet)
    - behavior tree wrapper
2. classes and methods to evolve BTs
    - mutation methods for all the possible behavior nodes
    - randomization methods for all nodes
    - recombination for the behavior nodes that may need that (not action nodes)
3. a class that manages the evolutionary aspects
    - selction strategy
    - recombination strategy
    - mutation strategy

## The Platform

## Evolution in BTs

## Differens Strategies for Evolution

Finally, we implemented a class to manage evolution of the population of behavior trees, these are the main functionalities:

1.  generate initial population
2.  get evaluated population after each episode and refine evaluation
3.  evolve the population
4.  return the evolved population

To generate the initial population we can adopt two strategies:

-   create the population randomly
-   include in the population a list of pre-trained trees (or even hadcrafted trees), in this second case we are pushing towards exploitation, and, therefore, we may loose some well-performing models.

The evaluation of the population is performed mainly by the environment of the game, which has its own reward system.
In our implementation it is possible to specify also the parameter `no_big_trees` which is used to obtain a debuff on the fitness of big behavior trees: it is multiplied by the size of the tree (depth or nr. of nodes) and then subtracted from the tree's fitness.

The evolution of the population is performed ...
