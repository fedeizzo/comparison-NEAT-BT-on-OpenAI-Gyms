from ftplib import all_errors
import json
from composite_nodes import composite_node_classes, CompositeNode
from behavior_node import BehaviorNode, InputIndex, BehaviorNodeTypes
from action_nodes import action_node_classes
import numpy as np
import pickle
import random

name_to_class = {
    cl.__name__: cl for cl in (composite_node_classes + action_node_classes)
}


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

    def to_json(self, filename):
        """Saves the tree into a json file in almost human-readable format.
        For each node it saves:
        - class
        - parameters
        - children

        Args:
            filename (str): name of the json file.
        """
        all_nodes = {}
        fifo: list[tuple[str, BehaviorNode]] = list()
        fifo.append(("0", self.root))
        node_global_index = 1
        while len(fifo):
            index, node = fifo.pop(0)
            all_nodes[index] = {}
            all_nodes[index]["class"] = node.__class__.__name__
            all_nodes[index]["parameters"] = node.parameters
            all_nodes[index]["children"] = []
            if hasattr(node, "children"):
                for child in node.children:
                    all_nodes[index]["children"].append(node_global_index)
                    fifo.append((str(node_global_index), child))
                    node_global_index += 1
        with open(filename, "w") as outfile:
            json.dump(all_nodes, outfile)

    @staticmethod
    def from_json(filename):
        """Creates a BT give a certain json which specifies the strucutre of
        the nodes. The format of the json must be the same as returned from
        method to_json().

        Args:
            filename (str): path to the file containing the json description.
        """
        with open(filename, "r") as infile:
            json_description = json.load(infile)
        # first, create all the nodes
        all_nodes = {}
        relations = {}
        for index in json_description.keys():
            node_class = name_to_class[json_description[index]["class"]]
            node_parameters = json_description[index]["parameters"]
            node = node_class(parameters=node_parameters)
            node_children = json_description[index]["children"]
            if len(node_children):
                relations[index] = node_children
            all_nodes[index] = node
        # then append children to the composite nodes
        for composite_idx in relations.keys():
            children_idxs = relations[composite_idx]
            for child_idx in children_idxs:
                all_nodes[composite_idx].insert_child(all_nodes[str(child_idx)])
        # finally, create a new bt
        new_bt = BehaviorTree()
        # and set the root
        new_bt.root = all_nodes["0"]
        return new_bt


if __name__ == "__main__":
    bt = BehaviorTree.generate(2)
    print(bt)
    bt.to_json("try.json")
    print("===============")
    loaded = BehaviorTree.from_json("try.json")
    print(loaded)
    # sample_input = np.zeros((64))
    # sample_input[InputIndex.Ability0Ready] = 1
    # sample_input[InputIndex.Ability1Ready] = 1
    # sample_input[InputIndex.Ability2Ready] = 1

    # result = bt.tick(sample_input)
    # import pdb

    # pdb.set_trace()
    # print(f"result: {result}")

    # bt1 = BehaviorTree.generate(2)
    # bt2 = BehaviorTree.generate(2)
    # print(bt1)
    # print(bt2)
    # bt1.recombination(bt2)
    # print("========= recombination =========")
    # print(bt1)
    # print(bt2)
