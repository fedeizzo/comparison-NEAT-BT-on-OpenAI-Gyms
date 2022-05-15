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

We created a platform for the architecture of BTs which is specialized to this peculiar task. We thought to create it from zero because the task is pretty specific and because we need full control of the low level classes of each node to implement evolutionary capabilities.

## Evolution in BTs

The basic classes of behavior trees have been extended with capabilities needed for th evolutionary approach: every node presents the method mutate and randomize.

The method randomize returns a randomized version of the same class of node, the method mutate randomly mutates the node.
Composite nodes are mutated by mutating the order of the children, adding or removing a child and propagating the mutation to the subtree.

The leaf nodes mutate by just mutating their parameters.

The class of the behavior tree (wrapper) exposes also one method to perform recombination, by swapping subtrees between two different trees.

## Differens Strategies for Evolution

Finally, we implemented a class to manage evolution of the population of behavior trees, these are the main functionalities:

1.  generate initial population
2.  get evaluated population after each episode and refine evaluation
3.  evolve the population
4.  return the evolved population

### 1. Generation of the new population

To generate the initial population we can adopt two strategies:

-   create the population randomly
-   [not implemented yet] include in the population a list of pre-trained trees (or even hadcrafted trees), in this second case we are pushing towards exploitation, and, therefore, we may loose some well-performing models.

### 2. Evaluation refinement

The evaluation of the population is performed mainly by the environment of the game, which has its own reward system.
In our implementation it is possible to specify also the parameter `no_big_trees` which is used to obtain a debuff on the fitness of big behavior trees: it is multiplied by the size of the tree (depth or nr. of nodes) and then subtracted from the tree's fitness.

This step is performed after the "real" evaluation from the game engine.

### 3. Evolution of the population

In the module `behavior_tree_evolution.py` we can find all the logic of the evolution. This means basically the following flow of operations:

1. selection of the pool from which to extract the parents following the given mu lambda strategy (comma or plus)
2. selection of the parents from the given pool
3. creation of the new population with elitism (if set)
4. completion of the new population by mutation or crossover of the parents

All these steps are monitored by saving the values of fitness (min, mean, max), depth and size (average, max).
