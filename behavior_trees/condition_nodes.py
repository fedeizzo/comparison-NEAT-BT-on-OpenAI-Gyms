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

class CheckConditionGreater(ConditionNode):

    def __init__(self, parameters):
        super().__init__(parameters)

    def applicable(self, input):
        # always applicable to check conditions
        return True

    def run(self, input):
        # check if input is greater than condition
        value = InputIndex.get_value(self.parameters["index"], input)
        if value > self.parameters["condition"]:

    @staticmethod
    def get_random_node():
        index = randint(0,len(InputIndex)-1)
        name = next(name for name, value in vars(InputIndex).items() if value == InputIndex)
        input_properties = getattr(InputProperties,name)
        condition = random()*(input_properties['max']-input_properties['min'])+input_properties['min']
        parameters = {"input_name":name,"input_index":index,"condition": condition,}
        return CheckConditionGreater(parameters)
    def mutate(self, prob: float):
        pass
