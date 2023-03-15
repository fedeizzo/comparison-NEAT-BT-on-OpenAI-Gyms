import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from random import randint, random

from behavior_node import (BehaviorNode, BehaviorNodeTypes,
                           BehaviorStates, InputIndex, OutputIndex)


class ActionNode(BehaviorNode):
    """Action nodes perform a single action.
    They return as state RUNNING.
    """

    def __init__(self, parameters, ticks_to_run: int=1):
        super().__init__(BehaviorNodeTypes.ACT, parameters)
        self.ticks_to_run = ticks_to_run
        self.max_ticks_to_run = ticks_to_run

    def copy(self):
        self_class = self.__class__
        print(self_class)
        copy = self_class(self.parameters)
        return copy

    def get_size(self):
        """Returns a tuple (depth,count) where depth is the level of the node
        starting from the leaves, and count is the count of nodes below+this 
        node.
        """
        return (1, 1)


class MoveNode(ActionNode):
    "Action node that moves the Derks front or back"

    def __init__(self, parameters):
        super().__init__(parameters, ticks_to_run=1)

    def applicable(self, input):
        """Always return true."""
        return True

    def run(self, input):
        action = np.zeros((5,))

        if self.ticks_to_run == self.max_ticks_to_run:
            self.ticks_to_run -= 1
            action[OutputIndex.MoveX] = self.parameters["move_x"]
            return BehaviorStates.RUNNING, action
        
        if self.ticks_to_run == 0:
            self.ticks_to_run = self.max_ticks_to_run
            return BehaviorStates.SUCCESS, action
        else:
            self.ticks_to_run -= 1
            return BehaviorStates.RUNNING, action

    @staticmethod
    def get_random_node():
        parameters = {"move_x": random() * 2 - 1}
        return MoveNode(parameters)

    def mutate(self, prob: float, all_mutations):
        """Mutates the focusing ability with probability prob.
        Args:
            prob (float): probability of mutation.
        """
        if random() < prob:
            self.parameters["move_x"] = random() * 2 - 1


class RotateNode(ActionNode):
    "Action node that rotates the Derks left or right"

    def __init__(self, parameters):
        super().__init__(parameters, ticks_to_run=1)

    def applicable(self, input):
        """Always return true."""
        return True
    
    def run(self, input):
        action = np.zeros((5,))

        if self.ticks_to_run == self.max_ticks_to_run:
            self.ticks_to_run -= 1
            action[OutputIndex.Rotate] = self.parameters["rotate"]
            return BehaviorStates.RUNNING, action
        
        if self.ticks_to_run == 0:
            self.ticks_to_run = self.max_ticks_to_run
            return BehaviorStates.SUCCESS, action
        else:
            self.ticks_to_run -= 1
            return BehaviorStates.RUNNING, action

    @staticmethod
    def get_random_node():
        parameters = {"rotate": random() * 2 - 1}
        return RotateNode(parameters)

    def mutate(self, prob: float, all_mutations):
        """Mutates the focusing ability with probability prob.
        Args:
            prob (float): probability of mutation.
        """
        if random() < prob:
            self.parameters["rotate"] = random() * 2 - 1


class CastNode(ActionNode):
    """Action node that casts one of the three abilities."""

    def __init__(self, parameters):
        super().__init__(parameters, ticks_to_run=1)

    def applicable(self, input):
        """Checks if the ability is ready to be cast.

        Args:
            input (np.ndarray): observations input array.

        Returns:
            boolean: True if the ability is ready, False otherwise.
        """
        if not bool(input[InputIndex.HasFocus]):
            return False

        # watch out, the input indexes ability with -1 w.r.t. the actual ability
        ability_name = f"Ability{self.parameters['cast_ability']-1}Ready"
        return input[int(InputIndex.__getitem__(ability_name))]

    def run(self, input):
        action = np.zeros((5,))

        if self.ticks_to_run == self.max_ticks_to_run:
            self.ticks_to_run -= 1
            action[OutputIndex.CastingSlot] = self.parameters["cast_ability"]
            return BehaviorStates.RUNNING, action
        
        if self.ticks_to_run == 0:
            self.ticks_to_run = self.max_ticks_to_run
            return BehaviorStates.SUCCESS, action
        else:
            self.ticks_to_run -= 1
            return BehaviorStates.RUNNING, action

    @staticmethod
    def get_random_node():
        # players can be 1, 2 or 3
        parameters = {"cast_ability": randint(1, 3)}
        return CastNode(parameters)

    def mutate(self, prob: float, all_mutations):
        """Mutates the casting ability with probability prob.

        Args:
            prob (float): probability of mutation.
        """
        if random() < prob:
            # players can be 1, 2 or 3
            self.parameters["cast_ability"] = randint(1, 3)


class ChangeFocusNode(ActionNode):
    """Action node that changes the focus of the derkling."""

    def __init__(self, parameters):
        super().__init__(parameters, ticks_to_run=2)

    def applicable(self, input):
        """Always true."""
        return True

    def run(self, input):
        """Returns an action with ChangeFocus set to self.parameters["focus"].
        #! may be interesting to return SUCCESS instead of RUNNING

        Args:
            input (np.ndarray): observations input array.

        Returns:
            np.ndarray: action returned.
        """
        action = np.zeros((5,))

        # print(self.ticks_to_run, self.max_ticks_to_run, bool(input[InputIndex.HasFocus]))
        # import pdb; pdb.set_trace()

        if self.ticks_to_run == self.max_ticks_to_run:
            self.ticks_to_run -= 1
            action[OutputIndex.ChangeFocus] = self.parameters["focus"]
            return BehaviorStates.RUNNING, action
        
        if self.ticks_to_run == 0:
            self.ticks_to_run = self.max_ticks_to_run
            return BehaviorStates.SUCCESS, action
        else:
            self.ticks_to_run -= 1
            return BehaviorStates.RUNNING, action

    @staticmethod
    def get_random_node():
        parameters = {"focus": randint(0, 7)}
        return ChangeFocusNode(parameters)

    def mutate(self, prob: float, all_mutations):
        """Mutates the focusing ability with probability prob.

        Args:
            prob (float): probability of mutation.
        """
        if random() < prob:
            self.parameters["focus"] = randint(0, 7)


class ChaseFocusNode(ActionNode):
    """Action node that makes the derkling move in the direction of its focus."""

    def __init__(self, parameters):
        super().__init__(parameters, ticks_to_run=1)

    def applicable(self, input):
        """Checks if the derkling can chase its focus."""
        return bool(input[InputIndex.HasFocus])

    def run(self, input):
        """Returns an action with ChangeFocus set to self.parameters["chase_focus"].

        Args:
            input (np.ndarray): observations input array.

        Returns:
            np.ndarray: action returned.
        """
        action = np.zeros((5,))

        if self.ticks_to_run == self.max_ticks_to_run:
            self.ticks_to_run -= 1
            action[OutputIndex.ChaseFocus] = self.parameters["chase_focus"]
            return BehaviorStates.RUNNING, action
        
        if self.ticks_to_run == 0:
            self.ticks_to_run = self.max_ticks_to_run
            return BehaviorStates.SUCCESS, action
        else:
            self.ticks_to_run -= 1
            return BehaviorStates.RUNNING, action
    @staticmethod
    def get_random_node():
        parameters = {"chase_focus": random()}
        return ChaseFocusNode(parameters)

    def mutate(self, prob: float, all_mutations):
        """Mutates the focusing ability with probability prob.

        Args:
            prob (float): probability of mutation.
        """
        if random() < prob:
            self.parameters["chase_focus"] = random()


action_node_classes = [
    # MoveNode,
    # RotateNode,
    CastNode,
    ChangeFocusNode,
    ChaseFocusNode,
]


if __name__ == "__main__":
    sample_input = np.zeros((64,))
    sample_input[InputIndex.Ability1Ready] = 1
    sample_input[InputIndex.HasFocus] = 1
    for clas in action_node_classes:
        node = clas.get_random_node()
        print(node)

        # node.mutate(1)
        # print(node)

        result = node.tick(sample_input)
        print(result)
