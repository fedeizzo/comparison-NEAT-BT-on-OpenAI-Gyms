import random
from typing import Optional

import numpy as np
from action_nodes import action_node_classes
from behavior_node import BehaviorNode, BehaviorNodeTypes, BehaviorStates
from condition_nodes import condition_node_classes

"""
May implement decorators.
Not sure that we need them... only inverter?

#! implement a maximum depth limit for node generation
"""


class CompositeNode(BehaviorNode):
    def __init__(self, type, parameters):
        super().__init__(type, parameters)
        # composite nodes do have children
        self.children: list[BehaviorNode] = []
        self.last_child_ticked = 0

    def insert_child(self, child: BehaviorNode, position: Optional[int]=None):
        if position is None:
            position = len(self.children)
        self.children.insert(position, child)

    def remove_child(self, position):
        return self.children.pop(position)

    def mutate(self, prob: int=0.2, all_mutations: bool=False):
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
        if (all_mutations or mutation_type == 1) and random.random() < prob and len(self.children) > 1:
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

    def __str__(self, indent:int=0) -> str:
        string_form = super().__str__(indent)
        for child in self.children:
            child_str = child.__str__(indent + 1)
            string_form += "\n" + child_str
        return string_form

    def copy(self):
        """Manual implementation of deepcopy.
        """
        self_class = self.__class__
        copy = self_class(self.parameters)
        copy.children = []
        for child in self.children:
            copy.children.append(child.copy())
        return copy

    def get_size(self):
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
        return (depth, count+1)


class SequenceNode(CompositeNode):
    """I decide to implement the sequence node as node with memory in this
    case: the node will execute each child once and then remember, on the next
    tick where it ended and start again from that point.
    """

    def __init__(self, parameters={}):
        super().__init__(BehaviorNodeTypes.SEQ, parameters)
        self.last_child_ticked = 0

    def applicable(self, input):
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
        result = (BehaviorStates.SUCCESS, np.zeros(5,))

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
                # wait for the next derk environment iteration to continue
                break
            elif result[0] == BehaviorStates.SUCCESS:  # condition nodes are always either success or failure
                continue
            elif result[0] == BehaviorStates.PARTIAL:
                self.last_child_ticked -= 1
                break

        # ticked all children: restart from 0
        if self.last_child_ticked == len(self.children):
            self.last_child_ticked = 0
        else:
            if result[0] == BehaviorStates.RUNNING:
                result = (BehaviorStates.PARTIAL, result[1])

        return result

    @staticmethod
    def get_random_node(num_children=1):
        """Generate a random instance of the SequenceNode."""
        sequence = SequenceNode()
        for _ in range(num_children):
            child_class = np.random.choice(candidate_classes)
            child = child_class.get_random_node()
            sequence.insert_child(child)
        return sequence


class FallbackNode(CompositeNode):
    """Fallback node with memory. It executes all children in order until one succeeds."""

    def __init__(self, parameters={}):
        super().__init__(BehaviorNodeTypes.FALL, parameters)
        self.last_child_ticked = 0

    def applicable(self, input):
        """Always true."""
        return True

    def run(self, input):
        """Runs the children in order as in self.children.
        It has memory of the last child run.

        Args:
            input (np.ndarray): observations input array.
        """
        result = (BehaviorStates.SUCCESS, np.zeros(5))
        for i in range(self.last_child_ticked, len(self.children)):
            self.last_child_ticked += 1
            result = self.children[i].tick(input)
            # print(self.__str__(0))
            # print(result)
            # print(i, self.last_child_ticked)
            # import pdb; pdb.set_trace()
            if (
                result[0] == BehaviorStates.SUCCESS
                or result[0] == BehaviorStates.RUNNING
            ):
                self.last_child_ticked = 0
                break
            elif result[0] == BehaviorStates.FAILURE:
                continue
            elif result[0] == BehaviorStates.PARTIAL:
                self.last_child_ticked -= 1
                break

        # ticked all children: restart from 0
        if self.last_child_ticked == len(self.children):
            self.last_child_ticked = 0
        return result

    @staticmethod
    def get_random_node(num_children=2):
        """Generate a random instance of the BehaviorNode."""
        fallback = FallbackNode()
        for _ in range(num_children):
            child_class: BehaviorNode = np.random.choice(candidate_classes)
            child = child_class.get_random_node()
            fallback.insert_child(child)
        return fallback


composite_node_classes = [SequenceNode, FallbackNode]

candidate_classes = action_node_classes + composite_node_classes + condition_node_classes


if __name__ == "__main__":
    # sn = SequenceNode.get_random_node()
    # sn.insert_child(CastNode.get_random_node(), len(sn.children))
    # sn.insert_child(CastNode.get_random_node(), len(sn.children))
    # for _ in range(3):
    #     sn.mutate(0.5)
    #     print(sn)

    fb = FallbackNode.get_random_node(num_children=3)
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
