import itertools
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import numpy as np


class BehaviorStates(Enum):
    SUCCESS = 0
    RUNNING = 1
    FAILURE = 2


class BehaviorNodeTypes(Enum):
    ACT = 0
    SEQ = 1
    FALL = 2
    COND = 3
    DEC = 4  # ? will use decorators?


class BehaviorNode(ABC):
    BEHAVIOR_NODE_ID = itertools.count()

    def __init__(self, type, parameters):
        self.type = type
        self.parameters = parameters
        self.status = None
        self.id = next(BehaviorNode.BEHAVIOR_NODE_ID)

    def tick(self, input):
        """Launches the tick to the node, it implements the standard routine
        with the check of applicable and the call to method run afterwards.

        Args:
            input (np.ndarray): observations input array.

        Returns:
            tuple[BehaviorStates, np.ndarray]: the result of the run method is
            a tuple with the return state (SUCCESS, RUNNING or FAILURE) and the
            action to perform.
        """
        if self.applicable(input):
            result = self.run(input)
        else:
            result = (BehaviorStates.FAILURE, np.zeros(shape=(5,)))
        self.status = result[0]
        return result

    @abstractmethod
    def applicable(self, input):
        """Check if the node run is executable.

        Args:
            input (np.ndarray): observations input array.
        """
        pass

    @abstractmethod
    def run(self, input) -> Tuple[BehaviorStates, np.ndarray]:
        """Runs the node.
        In composite nodes it will tick the children and implement the logic of
        the switch between the various children and the combination of returns.

        In action nodes it will return the action and the state.

        In condition nodes it will return the result of the condition (the
        state) and an empty action.

        Args:
            input (np.ndarray): observations input array.
        """
        ...

    @staticmethod
    @abstractmethod
    def get_random_node():
        """Generate a random instance of the BehaviorNode."""
        pass

    @abstractmethod
    def mutate(self, prob, all_mutations = False):
        """Randomly mutates the node with probability prob.

        Args:
            prob (float): probability of the mutation, between 0 and 1.
        """
        pass

    def __str__(self, indent=0) -> str:
        string_form = "\t" * indent
        string_form += f"{self.__class__.__name__}#{self.id}\tparams: {self.parameters}"
        return string_form

    def is_regular(self):
        """Checks if the tree is regular, namely if it does not contain two 
        times the very same node.
        """
        tree_nodes: set[BehaviorNode] = set()
        nodes = [self]
        while len(nodes) > 0:
            node = nodes.pop()
            if hasattr(node, "children"):
                for child in node.children:
                    nodes.append(child)
            if node.id in tree_nodes:
                return False
            tree_nodes.add(node.id)
        return True

    @abstractmethod
    def copy(self):
        """Manual implementation of deepcopy.
        """
        pass

    @abstractmethod
    def get_size(self):
        """Returns a tuple (depth,count) where depth is the level of the node
        starting from the leaves, and count is the count of nodes below+this 
        node.
        """


if __name__ == "__main__":
    print(BehaviorStates.SUCCESS)
    print(BehaviorStates.RUNNING)
    print(BehaviorStates.FAILURE)
    # print(InputIndex.__getitem__("Ability0Ready"))
