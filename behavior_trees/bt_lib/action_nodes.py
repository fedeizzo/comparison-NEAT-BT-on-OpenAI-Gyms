from bt_lib.behavior_node import BehaviorNode, BehaviorNodeTypes


class ActionNode(BehaviorNode):
    """Action nodes perform a single action.
    They return as state RUNNING.
    """

    def __init__(self, parameters: dict, ticks_to_run: int = 1):
        super().__init__(BehaviorNodeTypes.ACT, parameters)
        self.ticks_to_run = ticks_to_run
        self.max_ticks_to_run = ticks_to_run

    def copy(self) -> "ActionNode":
        self_class = self.__class__
        print(self_class)
        copy = self_class(self.parameters)
        return copy

    def get_size(self) -> tuple[int, int]:
        """Returns a tuple (depth,count) where depth is the level of the node
        starting from the leaves, and count is the count of nodes below+this
        node.
        """
        return (1, 1)
