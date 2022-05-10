from action_nodes import ActionNode, action_node_classes, CastNode
from behavior_node import *
import random

"""Maybe, implement decorators.
Not sure that we need them... only inverter?
"""


class CompositeNode(BehaviorNode):
    def __init__(self, type, parameters):
        super().__init__(type, parameters)
        # composite nodes do have children
        self.children: list[BehaviorNode] = list()

    def insert_child(self, child: BehaviorNode, position: int = -1):
        if position == -1:
            position = len(self.children)
        self.children.insert(position, child)

    def remove_child(self, position):
        return self.children.pop(position)

    def mutate(self, prob=0.2):
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

        # add a node with probability prob
        if random.random() < prob:
            candidate_classes = action_node_classes + composite_node_classes
            child_class: BehaviorNode = np.random.choice(candidate_classes)
            child = child_class.get_random_node()
            self.insert_child(child, len(self.children))

        # remove a node with probability prob
        if random.random() < prob and len(self.children) > 1:
            removing_index = random.randint(0, (len(self.children) - 1))
            if removing_index > self.last_child_ticked:
                self.last_child_ticked -= 1
            # print("remove")
            self.remove_child(removing_index)

        # swap children
        if random.random() < prob:
            # print("swap")
            np.random.shuffle(self.children)
            self.last_child_ticked = 0

        # mutate child
        for i in range(len(self.children)):
            if random.random() < prob:
                self.children[i].mutate(prob)

    def __str__(self, indent=0) -> str:
        string_form = super().__str__(indent)
        for child in self.children:
            child_str = child.__str__(indent + 1)
            string_form += "\n" + child_str
        return string_form


class SequenceNode(CompositeNode):
    """I decide to implement the sequence node as node with memory in this
    case: the node will execute each child once and then remember, on the next
    tick where it ended and start again from that point.
    """

    def __init__(self, parameters={}) -> None:
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

    def __init__(self, parameters={}) -> None:
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
    def get_random_node(num_children=1):
        """Generate a random instance of the BehaviorNode."""
        fallback = FallbackNode()
        candidate_classes = action_node_classes + composite_node_classes
        for _ in range(num_children):
            child_class: BehaviorNode = np.random.choice(candidate_classes)
            child = child_class.get_random_node()
            fallback.insert_child(child, -1)
        return fallback


composite_node_classes = [SequenceNode, FallbackNode]


if __name__ == "__main__":
    # sn = SequenceNode.get_random_node()
    # sn.insert_child(CastNode.get_random_node(), len(sn.children))
    # sn.insert_child(CastNode.get_random_node(), len(sn.children))
    # for _ in range(3):
    #     sn.mutate(0.5)
    #     print(sn)

    fb = FallbackNode.get_random_node()
    fb.insert_child(CastNode.get_random_node(), len(fb.children))
    # fb.insert_child(CastNode.get_random_node(), len(fb.children))
    # for _ in range(3):
    #     fb.mutate(0.5)
    print(fb)
    sample_input = np.zeros((64))
    sample_input[InputIndex.Ability0Ready] = 0
    sample_input[InputIndex.Ability1Ready] = 0
    sample_input[InputIndex.Ability2Ready] = 1
    result = fb.tick(sample_input)
    print(result)
