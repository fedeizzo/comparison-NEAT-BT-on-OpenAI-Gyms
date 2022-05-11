# Behavior Trees Evolution

Here we report the results of the experiments and the discussions deveoped during the impementation of the BT architecture.

## The Platform

We developed and braind new BT platform for BTs specialized for this very task. This platform offers the following features:
1. classes and methods for the basic structures of BT
    - action nodes
    - composite nodes
    - condition nodes (not yet)
    - behavior tree wrapper
2. classes and methods to evolve BTs
    - mutation methods for all the possible behavior nodes
    - recombination for the behavior nodes that may need that
    