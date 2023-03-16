import os
import random
import sys
from typing import Optional

import numpy as np
from bt_lib.action_nodes import ActionNode
from bt_lib.behavior_node import BehaviorNode, BehaviorNodeTypes, BehaviorStates
from bt_lib.condition_nodes import ConditionNode

"""
May implement decorators.
Not sure that we need them... only inverter?

#! implement a maximum depth limit for node generation
"""


class CompositeNode(BehaviorNode):
    def __init__(self, type: BehaviorNodeTypes, parameters: dict):
        super().__init__(type, parameters)
        # composite nodes do have children
        self.children: list[BehaviorNode] = []
        self.last_child_ticked = 0

    def insert_child(self, child: BehaviorNode, position: Optional[int] = None):
        if position is None:
            position = len(self.children)
        self.children.insert(position, child)

    def remove_child(self, position: int):
        return self.children.pop(position)

    def mutate(
        self,
        candidate_classes: list[type[BehaviorNode]],
        prob: float = 0.2,
        all_mutations: bool = False,
    ):
        """Mutates the sequence node.

        Possible mutations:
        1. addition of a node
        2. removal of a node (check at least one child)
        3. change order of the nodes
        4. call mutate on all children

        Possible improvement: mutation probability for children may be divided
        by the number of children.

        Args:
            prob (float): probability of the mutation, between 0 and 1.
            all_mutations (bool): perform all possible type of mutations.
        """
        self.last_child_ticked = 0
        mutation_type = random.randint(0, 2)

        # add a node with probability prob
        if (all_mutations or mutation_type == 0) and random.random() < prob:
            child_class: BehaviorNode = np.random.choice(candidate_classes)
            child = child_class.get_random_node()
            self.insert_child(child)

        # remove a node with probability prob
        if (
            (all_mutations or mutation_type == 1)
            and random.random() < prob
            and len(self.children) > 1
        ):
            removing_index = random.randint(0, (len(self.children) - 1))
            # no need to update last child ticked
            # if the tree is reset at each play
            self.remove_child(removing_index)

        # swap children
        if (all_mutations or mutation_type == 2) and random.random() < prob:
            np.random.shuffle(self.children)

        # mutate all children
        for c in self.children:
            c.mutate(prob, all_mutations)

    def __str__(self, indent: int = 0) -> str:
        string_form = super().__str__(indent)
        for child in self.children:
            child_str = child.__str__(indent + 1)
            string_form += "\n" + child_str
        return string_form

    def copy(self) -> "CompositeNode":
        """Manual implementation of deepcopy."""
        self_class = self.__class__
        copy = self_class(self.parameters)
        copy.children = []
        for child in self.children:
            copy.children.append(child.copy())
        return copy

    def get_size(self) -> tuple[int, int]:
        """Returns a tuple (depth,count) where depth is the level of the node
        starting from the leaves, and count is the count of nodes below+this
        node.
        """
        depth = -1
        count = 0
        for child in self.children:
            c_depth, c_count = child.get_size()
            count += c_count
            depth = max(depth, c_depth)
        return (depth, count + 1)


class SequenceNode(CompositeNode):
    """I decide to implement the sequence node as node with memory in this
    case: the node will execute each child once and then remember, on the next
    tick where it ended and start again from that point.
    """

    def __init__(self, parameters: dict = {}):
        super().__init__(BehaviorNodeTypes.SEQ, parameters)
        self.last_child_ticked = 0

    def applicable(self, input: np.ndarray) -> bool:
        """Always true."""
        return True

    def run(self, input):
        """Runs the children in order as in self.children.
        It has memory of the last child run.

        There is no need to manage addition or deletion of nodes at runtime
        since these operations are performed off-game.

        Args:
            input (np.ndarray): observations input array.
        """
        result = (BehaviorStates.SUCCESS, None)

        for i in range(self.last_child_ticked, len(self.children)):
            self.last_child_ticked += 1
            result = self.children[i].tick(input)

            # print(self.__str__(0))
            # print(result)
            # print(i, self.last_child_ticked)
            # import pdb; pdb.set_trace()

            if result[0] == BehaviorStates.FAILURE:
                self.last_child_ticked = 0
                break
            elif result[0] == BehaviorStates.RUNNING:
                self.last_child_ticked -= 1
                break
            elif result[0] == BehaviorStates.SUCCESS:
                continue

        # ticked all children: restart from 0
        if self.last_child_ticked == len(self.children):
            self.last_child_ticked = 0

        return result

    @staticmethod
    def get_random_node(
        action_node_classes: list[type[ActionNode]],
        condition_node_classes: list[type[ConditionNode]],
        composite_node_classes: list[type[CompositeNode]],
        num_children: int = 1,
    ) -> "SequenceNode":
        """Generate a random instance of the SequenceNode.
        Args:
            candidate_classes (list[BehaviorNode]): list of classes to choose from.
            num_children (int): number of children to generate.
        """
        sequence = SequenceNode()
        for _ in range(num_children):
            child_class: BehaviorNode = np.random.choice(
                action_node_classes + condition_node_classes + composite_node_classes
            )

            # only composite nodes require class types when generating random nodes
            if child_class in composite_node_classes:
                child = child_class.get_random_node(
                    action_node_classes,
                    condition_node_classes,
                    composite_node_classes,
                    num_children,
                )
            else:
                child = child_class.get_random_node()
            sequence.insert_child(child)
        return sequence


class FallbackNode(CompositeNode):
    """Fallback node with memory. It executes all children in order until one succeeds."""

    def __init__(self, parameters: dict = {}):
        super().__init__(BehaviorNodeTypes.FALL, parameters)
        self.last_child_ticked = 0

    def applicable(self, input: np.ndarray) -> bool:
        """Always true."""
        return True

    def run(self, input: np.ndarray) -> tuple[BehaviorStates, np.ndarray]:
        """Runs the children in order as in self.children.
        It has memory of the last child run.

        Args:
            input (np.ndarray): observations input array.
        """
        result = (BehaviorStates.SUCCESS, None)
        for i in range(self.last_child_ticked, len(self.children)):
            self.last_child_ticked += 1
            result = self.children[i].tick(input)

            # print(self.__str__(0))
            # print(result)
            # print(i, self.last_child_ticked)
            # import pdb; pdb.set_trace()

            if result[0] == BehaviorStates.SUCCESS:
                self.last_child_ticked = 0
                break
            elif result[0] == BehaviorStates.RUNNING:
                self.last_child_ticked -= 1
                break
            elif result[0] == BehaviorStates.FAILURE:
                continue

        # ticked all children: restart from 0
        if self.last_child_ticked == len(self.children):
            self.last_child_ticked = 0
        return result

    @staticmethod
    def get_random_node(
        action_node_classes: list[type[ActionNode]],
        condition_node_classes: list[type[ConditionNode]],
        composite_node_classes: list[type[CompositeNode]],
        num_children: int = 2,
    ) -> "FallbackNode":
        """Generate a random instance of the BehaviorNode.
        Args:
            candidate_classes (list[BehaviorNode]): list of classes to choose from.
            num_children (int): number of children to generate.
        """
        fallback = FallbackNode()
        for _ in range(num_children):
            child_class: BehaviorNode = np.random.choice(
                action_node_classes + condition_node_classes + composite_node_classes
            )
            # composite nodes require class types when generating random nodes
            if child_class in composite_node_classes:
                child = child_class.get_random_node(
                    action_node_classes,
                    condition_node_classes,
                    composite_node_classes,
                    num_children,
                )
            else:
                child = child_class.get_random_node()
            fallback.insert_child(child)
        return fallback


composite_node_classes = [SequenceNode, FallbackNode]


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from derk.action_nodes import action_node_classes
    from derk.condition_nodes import condition_node_classes

    candidate_classes = (
        action_node_classes + condition_node_classes + composite_node_classes
    )
    name_to_class = {cl.__name__: cl for cl in (candidate_classes)}

    fb = FallbackNode.get_random_node(
        action_node_classes,
        condition_node_classes,
        composite_node_classes,
        num_children=2,
    )

    # fb.insert_child(CastNode.get_random_node(), len(fb.children))
    # # fb.insert_child(CastNode.get_random_node(), len(fb.children))
    # # for _ in range(3):
    # #     fb.mutate(0.5)
    # print(fb)
    # sample_input = np.zeros((64))
    # sample_input[InputIndex.Ability0Ready] = 0
    # sample_input[InputIndex.Ability1Ready] = 0
    # sample_input[InputIndex.Ability2Ready] = 1
    # result = fb.tick(sample_input)
    # print(result)
    fbcopy = fb.copy()
    print(fb)
    print(fbcopy)
    sq = SequenceNode.get_random_node(
        action_node_classes,
        condition_node_classes,
        composite_node_classes,
        num_children=2,
    )

    print(sq)
