import gymnasium as gym 
import os 
import sys
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from behavior_trees.condition_nodes import ConditionNode
from behavior_trees.behavior_node import BehaviorStates
from behavior_trees.condition_nodes import ConditionType
from input_output_lunar_lander import LanderInputIndex, LanderInputProperties

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

        condition_type = ConditionType[self.parameters['condition_type']]
        if condition_type == ConditionType.EQUAL:
            if input_value == condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(1,)))
        if condition_type == ConditionType.GREATER:
            if input_value > condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(1,)))
        if condition_type == ConditionType.LESS:
            if input_value < condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(1,)))
        if condition_type == ConditionType.GREATER_EQUAL:
            if input_value >= condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(1,)))
        if condition_type == ConditionType.LESS_EQUAL:
            if input_value <= condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(1,)))
        if condition_type == ConditionType.NOT_EQUAL:
            if input_value != condition_value:
                return (BehaviorStates.SUCCESS, np.zeros(shape=(1,)))
        
        return (BehaviorStates.FAILURE, np.zeros(shape=(1,)))
    
    @staticmethod
    def get_random_node():
        index = random.randint(0, len(LanderInputIndex)-1)
        name = next(name for name, value in vars(LanderInputIndex).items() if value == index)
        input_properties = getattr(LanderInputProperties, name)

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
            index = random.randint(0, len(LanderInputIndex)-1)
            name = next(name for name, value in vars(LanderInputIndex).items() if value == index)
            input_properties = getattr(LanderInputProperties, name)

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
condition_node_classes = [CheckConditionNode,]
if __name__ == "__main__":
    # test
    node = CheckConditionNode.get_random_node()
    print (node.parameters)
    node.tick(np.zeros(shape=(8,)))
    print (node)