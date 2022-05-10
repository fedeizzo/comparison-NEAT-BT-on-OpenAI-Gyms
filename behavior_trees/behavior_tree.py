from composite_nodes import composite_node_classes, CompositeNode
from behavior_node import BehaviorNode, InputIndex, BehaviorNodeTypes
from action_nodes import action_node_classes
import numpy as np
import pickle
import random


class BehaviorTree:
    """Wrapper class for the behavior tree.
    Contains utility method to manage a single complete tree.
    """
    def __init__(self) -> None:
        root_class: BehaviorNode = np.random.choice(composite_node_classes)
        self.root: CompositeNode = root_class.get_random_node()

    def __str__(self) -> str:
        return self.root.__str__()

    def save(self, path):
        with open(path, "wb") as outfile:
            pickle.dump(self, outfile)

    def mutate(self, prob):
        """Start mutation from the root, then propagate."""
        self.root.mutate(prob)

    def recombination(self, other: "BehaviorTree"):
        """Recombination between two different behavior trees, it should be
        a swap between two subtrees.

        Rough implementation right now: swap random nodes, the depth is decided
        randomly, the type of swapped subtree is not filtered

        Args:
            other (BehaviorTree): the other behavior tree that we want to use
            for recombination.
        """
        exchange_point_a = self.root
        child_a = self.root
        index_a = False
        while child_a.type != BehaviorNodeTypes.ACT and random.random() < 0.5:
            exchange_point_a = child_a
            index_a = random.randint(0, len(exchange_point_a.children) - 1)
            child_a = exchange_point_a.children[index_a]

        exchange_point_b = other.root
        child_b = other.root
        index_b = False
        while child_b.type != BehaviorNodeTypes.ACT and random.random() < 0.5:
            exchange_point_b = child_b
            index_b = random.randint(0, len(exchange_point_b.children) - 1)
            child_b = exchange_point_b.children[index_b]

        # print(f"Sbstituting\n\t{child_a}, idx{index_a} with \n\t{child_b}, idx{index_b}")
        # actual swap of subtrees
        exchange_point_a.children[index_a] = child_b
        exchange_point_b.children[index_b] = child_a

    @staticmethod
    def generate(min_children=5):
        """Create a new behavior tree with at least min_children child nodes.
        #! Would be great to set a minimum depth instead of width.

        Args:
            min_children (int, optional): minimum number of children for the 
            root. Defaults to 5.

        Returns:
            BehaviorTree: the newly instantiated behavior tree.
        """
        bt = BehaviorTree()
        candidate_classes = action_node_classes + composite_node_classes
        for _ in range(min_children):
            child_class: BehaviorNode = np.random.choice(candidate_classes)
            child = child_class.get_random_node()
            bt.root.insert_child(child, len(bt.root.children))
        return bt

    def tick(self, input):
        return self.root.tick(input)

if __name__ == "__main__":
    bt = BehaviorTree.generate(4)
    print(bt)
    sample_input = np.zeros((64))
    sample_input[InputIndex.Ability0Ready] = 1
    sample_input[InputIndex.Ability1Ready] = 1
    sample_input[InputIndex.Ability2Ready] = 1

    result = bt.tick(sample_input)
    import pdb; pdb.set_trace()
    print(f"result: {result}")

    # bt1 = BehaviorTree.generate(2)
    # bt2 = BehaviorTree.generate(2)
    # print(bt1)
    # print(bt2)
    # bt1.recombination(bt2)
    # print("========= recombination =========")
    # print(bt1)
    # print(bt2)
