import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bt_lib.behavior_node import (BehaviorNode, BehaviorNodeTypes,
                                  BehaviorStates)
from bt_lib.condition_nodes import ConditionNode, ConditionType

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from input_output import InputIndex, InputProperties, OutputIndex


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
            return (BehaviorStates.FAILURE, np.zeros(shape=(5,)))

        condition_type = ConditionType[self.parameters['condition_type']]
        if condition_type == ConditionType.EQUAL:
            if input_value == condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        if condition_type == ConditionType.GREATER:
            if input_value > condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        if condition_type == ConditionType.LESS:
            if input_value < condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        if condition_type == ConditionType.GREATER_EQUAL:
            if input_value >= condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        if condition_type == ConditionType.LESS_EQUAL:
            if input_value <= condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        if condition_type == ConditionType.NOT_EQUAL:
            if input_value != condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(5,)))
        
        return (BehaviorStates.FAILURE, np.zeros(shape=(5,)))
    
    @staticmethod
    def get_random_node():
        index = random.randint(0, len(InputIndex)-1)
        name = next(name for name, value in vars(InputIndex).items() if value == index)
        input_properties = getattr(InputProperties, name)

        if input_properties['type'] == bool:
            condition_value = random.choice([True, False])
            condition_type = random.choice([ConditionType.EQUAL, ConditionType.NOT_EQUAL])
        elif input_properties['type'] == float:
            condition_value = random.uniform(input_properties['min'], input_properties['max'])
            # avoid equal condition for floats
            condition_type = random.choice([ConditionType.LESS, ConditionType.GREATER])

        return CheckConditionNode({
            "input_name": name,
            "input_index": index,
            "condition_type": condition_type.name,
            "condition_value": condition_value,
        })

    def mutate(self, prob: float, all_mutations):
        # TODO: reuse get_random_node
        if random.random() < prob:
            index = random.randint(0, len(InputIndex)-1)
            name = next(name for name, value in vars(InputIndex).items() if value == index)
            input_properties = getattr(InputProperties, name)

            if input_properties['type'] == bool:
                condition_value = random.choice([True, False])
                condition_type = random.choice([ConditionType.EQUAL, ConditionType.NOT_EQUAL])
            elif input_properties['type'] == float:
                condition_value = random.uniform(input_properties['min'], input_properties['max'])
                # avoid equal condition for floats
                condition_type = random.choice([ConditionType.LESS, ConditionType.GREATER])

            self.parameters = {
                "input_name": name,
                "input_index": index,
                "condition_type": condition_type.name,
                "condition_value": condition_value,
            }


condition_node_classes = [CheckConditionNode]

if __name__ == "__main__":
    sample_input = np.zeros((64,))
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
