from sre_constants import FAILURE, SUCCESS
from action_nodes import ActionNode
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
    def copy(self):
        """Manual implementation of deepcopy.
        """
        self_class = self.__class__
        copy = self_class(self.parameters)
        return copy
    def get_size(self):
        """Returns a tuple (depth,count) where depth is the level of the node
        starting from the leaves, and count is the count of nodes below+this 
        node.
        """
        return (1, 1)

class CheckConditionNode(ConditionNode):

    def __init__(self, parameters):
        super().__init__(parameters)

    def applicable(self, input):
        # always applicable to check conditions
        return True

    def run(self, input):
        # check if input is greater than condition
        condition_value = self.parameters["condition_value"]
        input_value = input[self.parameters["input_index"]]
        if input_value == 2:
            return (BehaviorStates.FAILURE,np.zeros(shape=(5,)))
        condition_type = self.parameters['condition_type']
        if condition_type == 'equality':
            if input_value == condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        if condition_type == 'greater':
            if input_value > condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        if condition_type == 'lesser':
            if input_value < condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        return (BehaviorStates.FAILURE, np.zeros(shape=(5,)))
    
    @staticmethod
    def get_random_node():
        index = randint(0,len(InputIndex)-1)
        name = next(name for name, value in vars(InputIndex).items() if value == index)
        input_properties = getattr(InputProperties,name)
        if input_properties['type'] == bool:
            condition_value = randint(0,1)
            condition_type = 'equality'
        else:
            condition_value = random()*(input_properties['max']-input_properties['min'])+input_properties['min']
            if randint(0,1):
                condition_type = 'lesser'
            else:
                condition_type = 'greater'
        parameters = {"input_name":name,"input_index":index,'condition_type':condition_type,"condition_value": condition_value,}
        return CheckConditionNode(parameters)
    def mutate(self, prob: float, all_mutations):
        if random()<prob:
            index = randint(0,len(InputIndex)-1)
            name = next(name for name, value in vars(InputIndex).items() if value == index)
            input_properties = getattr(InputProperties,name)
            if input_properties['type'] == bool:
                condition_value = randint(0,1)
                condition_type = 'equality'
            else:
                condition_value = random()*(input_properties['max']-input_properties['min'])+input_properties['min']
                if randint(0,1):
                    condition_type = 'less'
                else:
                    condition_type = 'greater'
            self.parameters = {"input_name":name,"input_index":index,'condition_type':condition_type,"condition_value": condition_value,}

condition_node_classes = [CheckConditionNode,]

if __name__ == "__main__":
    sample_input = np.zeros((64))
    sample_input[InputIndex.Ability1Ready] = 1
    sample_input[InputIndex.HasFocus] = 1
    random_condition = CheckConditionNode.get_random_node()
    print(random_condition)
    print(sample_input)
    a = random_condition.tick(sample_input)
    print(random_condition)
    print(a)
     
    copy_condition = random_condition.copy()
    
    print(copy_condition)
    
    copy_condition.mutate(0.5, True)
    
    print(copy_condition)
    print(random_condition)
