import random

import numpy as np
from bt_lib.action_nodes import ActionNode
from bt_lib.behavior_node import BehaviorStates
from frozen_lake.input_output import (
    LakeInputIndex,
    LakeInputProperties,
    LakeOutputIndex,
)


class LakeAction(ActionNode):
    """Action node that makes the lake move in a direction."""

    def __init__(self, parameters):
        super().__init__(parameters, ticks_to_run=1)

    def applicable(self, input):
        """Checks if the lake can do action"""
        return True

    def run(self, input):
        """Returns an action according to the action type.

        Args:
            input (np.ndarray): observations input array.

        Returns:
            np.ndarray: action returned.
        """
        action = self.parameters["lake_action"]

        if self.ticks_to_run == self.max_ticks_to_run:
            self.ticks_to_run -= 1
            action = self.parameters["lake_action"]
            return BehaviorStates.RUNNING, action

        if self.ticks_to_run == 0:
            self.ticks_to_run = self.max_ticks_to_run
            return BehaviorStates.SUCCESS, action
        else:
            self.ticks_to_run -= 1
            return BehaviorStates.RUNNING, action
        
    def mutate(self, prob: float, all_mutations):
        """Mutates the action with another random action.
        Args:
            prob (float): probability of mutation.
        """
        if random.random() < prob:
            self.parameters["lake_action"] = random.choice(list(LakeOutputIndex))
    @staticmethod
    def get_random_node():
        parameters = {"lake_action": random.choice(list(LakeOutputIndex))}
        return LakeAction(parameters)



action_node_classes = [LakeAction]


if __name__ == "__main__":
    sample_input = np.zeros((8,))
    sample_input[LakeInputIndex.angular_velocity] = 1
    sample_input[LakeInputIndex.right_ground_contact] = 1
    for clas in action_node_classes:
        node = clas.get_random_node()
        print(node)

        # node.mutate(1)
        # print(node)

        result = node.tick(sample_input)
        print(result)
        result = node.tick(sample_input)
        print(result)
