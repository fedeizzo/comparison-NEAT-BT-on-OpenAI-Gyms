from action_nodes import ActionNode, action_node_classes, CastNode
from behavior_node import *
import random
import copy

"""
May implement decorators.
Not sure that we need them... only inverter?

#! implement a maximum depth limit for node generation
"""


class CompositeNode(BehaviorNode):
    def __init__(self, type, parameters):
        super().__init__(type, parameters)
        # composite nodes do have children
        self.children: list[BehaviorNode] = list()
        self.last_child_ticked = 0

    def insert_child(self, child: BehaviorNode, position: int = -1):
        if position == -1:
            position = len(self.children)
        self.children.insert(position, child)

    def remove_child(self, position):
        return self.children.pop(position)

    def mutate(self, prob=0.2, all_mutations=False):
        """Mutates the sequence node.

        Possible mutations:
        - addition of a node
        - removal of a node (check at least one child)
        - change order of the nodes
        - call mutate on child

        Implementded:
        - all of them, just need to select which ones to use

        Possible improvement: mutation probability for children may be divided
        by the number of children.

        Args:
            prob (float): probability of the mutation, between 0 and 1.
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

        # mutate children
        for c in self.children:
            c.mutate(prob)

        #! remove
        # to_mutate = list()
        # for i in range(len(self.children)):
        #     if random.random() < prob:
        #         if self.children[i].type == BehaviorNodeTypes.ACT:
        #             self.children[i].mutate(prob)
        #         else:
        #             to_mutate.append(self.children[i])

        # for mutating in to_mutate:
        #     mutating.mutate(prob)

    def __str__(self, indent=0) -> str:
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
        result = (
            BehaviorStates.SUCCESS,
            np.zeros(
                5,
            ),
        )
        for i in range(self.last_child_ticked, len(self.children)):
            self.last_child_ticked += 1
            result = self.children[i].tick(input)
            if result[0] == BehaviorStates.FAILURE:
                self.last_child_ticked = 0
                break
            if result[0] == BehaviorStates.RUNNING:
                break
        # ticked all children: restart from 0
        if self.last_child_ticked == len(self.children):
            self.last_child_ticked = 0
        return result

    @staticmethod
    def get_random_node(num_children=1):
        """Generate a random instance of the BehaviorNode."""
        sequence = SequenceNode()
        candidate_classes = action_node_classes + composite_node_classes
        for _ in range(num_children):
            child_class: BehaviorNode = np.random.choice(candidate_classes)
            child = child_class.get_random_node()
            sequence.insert_child(child, -1)
        return sequence


class FallbackNode(CompositeNode):
    """Fallback node with memory"""

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
        result = (
            BehaviorStates.SUCCESS,
            np.zeros(
                5,
            ),
        )
        for i in range(self.last_child_ticked, len(self.children)):
            self.last_child_ticked += 1
            result = self.children[i].tick(input)
            if (
                result[0] == BehaviorStates.SUCCESS
                or result[0] == BehaviorStates.RUNNING
            ):
                break
        # ticked all children: restart from 0
        if self.last_child_ticked == len(self.children):
            self.last_child_ticked = 0
        return result

    @staticmethod
    def get_random_node(num_children=2):
        """Generate a random instance of the BehaviorNode."""
        fallback = FallbackNode()
        candidate_classes = action_node_classes + composite_node_classes
        for _ in range(num_children):
            child_class: BehaviorNode = np.random.choice(candidate_classes)
            child = child_class.get_random_node()
            fallback.insert_child(child)
        return fallback


composite_node_classes = [SequenceNode, FallbackNode]

candidate_classes = action_node_classes + composite_node_classes


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
