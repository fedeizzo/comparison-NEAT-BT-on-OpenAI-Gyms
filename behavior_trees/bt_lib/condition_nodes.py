from enum import Enum

from bt_lib.behavior_node import BehaviorNode, BehaviorNodeTypes

"""
Implement all condition check nodes.
There will be for enemy distance, enemy angle, distance from border and others.
Here massively use the input observations.

All condition nodes are the derived from ConditionNode class.
"""


class ConditionType(Enum):
    """Enum for the type of condition."""

    EQUAL = 1
    GREATER = 2
    LESS = 3
    GREATER_EQUAL = 4
    LESS_EQUAL = 5
    NOT_EQUAL = 6


class ConditionNode(BehaviorNode):
    """Condition nodes perform an atomic condition check.
    They return as state FAILURE or SUCCESS.
    """

    def __init__(self, parameters: dict):
        super().__init__(BehaviorNodeTypes.COND, parameters)

    def copy(self) -> "ConditionNode":
        """Manual implementation of deepcopy."""
        self_class = self.__class__
        copy = self_class(self.parameters)
        return copy

    def get_size(self) -> tuple[int, int]:
        """Returns a tuple (depth,count) where depth is the level of the node
        starting from the leaves, and count is the count of nodes below+this
        node.
        """
        return (1, 1)
