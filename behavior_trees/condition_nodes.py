import numpy
from behavior_node import *
from random import randint, random

"""
Implement all condition check nodes.
There will be for enemy distance, enemy angle, distance from border and others.
Here massively use the input observations.

All condition nodes are the derived from ConditionNode class.
"""

class ConditionNode(BehaviorNode):
    """Condition nodes perform an atomic condition check.
    They return as state FAILURE or SUCCESS.
    """

    def __init__(self, parameters):
        super().__init__(BehaviorNodeTypes.COND, parameters)

class Example(ConditionNode):

    def __init__(self, parameters):
        super().__init__(parameters)

    def applicable(self, input):
        pass

    def run(self, input):
        pass

    @staticmethod
    def get_random_node():
        pass

    def mutate(self, prob: float):
        pass
