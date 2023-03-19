import json
import os
import pickle
import random
import sys

import numpy as np
from bt_lib.action_nodes import ActionNode
from bt_lib.behavior_node import BehaviorNode, BehaviorNodeTypes, BehaviorStates
from bt_lib.composite_nodes import CompositeNode
from bt_lib.condition_nodes import ConditionNode


class BehaviorTree:
    """Wrapper class for the behavior tree.
    Contains utility method to manage a single complete tree.
    """

    def __init__(
        self,
        action_node_classes: list[type[ActionNode]],
        condition_node_classes: list[type[ConditionNode]],
        composite_node_classes: list[type[CompositeNode]],
    ):
        self.fitness = 0
        self.action_node_classes = action_node_classes
        self.condition_node_classes = condition_node_classes
        self.composite_node_classes = composite_node_classes

        root_class = np.random.choice(self.composite_node_classes)
        self.root: CompositeNode = root_class.get_random_node(
            self.action_node_classes,
            self.condition_node_classes,
            self.composite_node_classes,
            1,
        )

    def __str__(self) -> str:
        return self.root.__str__()

    def save(self, path: str):
        with open(path, "wb") as outfile:
            pickle.dump(self, outfile)

    def mutate(self, prob: float, all_mutations: bool = False):
        """Start mutation from the root, then propagate."""
        if isinstance(self.root, CompositeNode):
            self.root.mutate(
                self.action_node_classes,
                self.condition_node_classes,
                self.composite_node_classes,
                prob,
                all_mutations,
            )
        else:
            self.root.mutate(prob, all_mutations)

    def recombination(self, other: "BehaviorTree", prob: float) -> "BehaviorTree":
        """Recombination between two different behavior trees, it should be
        a swap between two subtrees.

        Rough implementation right now: swap random nodes, the depth is decided
        randomly, the type of swapped subtree is not filtered

        Args:
            other (BehaviorTree): the other behavior tree that we want to use
            for recombination.
        """
        # deep copy to avoid "data race" condition
        from copy import deepcopy

        parent_a = deepcopy(self)
        parent_b = deepcopy(other)

        # find exchange point in the first tree
        exchange_point_a = parent_a.root
        child_a = parent_a.root
        index_a = None
        while (
            child_a.type != BehaviorNodeTypes.ACT
            and child_a.type != BehaviorNodeTypes.COND
            and random.random() < prob
        ):
            exchange_point_a = child_a
            index_a = random.randint(0, len(exchange_point_a.children) - 1)
            child_a = exchange_point_a.children[index_a]

        # find exchange point in the second tree
        exchange_point_b = parent_b.root
        child_b = parent_b.root
        index_b = None
        while (
            child_b.type != BehaviorNodeTypes.ACT
            and child_b.type != BehaviorNodeTypes.COND
            and random.random() < prob
        ):
            exchange_point_b = child_b
            index_b = random.randint(0, len(exchange_point_b.children) - 1)
            child_b = exchange_point_b.children[index_b]

        # actual swap of subtrees
        # need to check the special case where the child is the root
        if index_a is None:
            parent_a.root = child_b
        else:
            exchange_point_a.children[index_a] = child_b

        if index_b is None:
            parent_b.root = child_a
        else:
            exchange_point_b.children[index_b] = child_a

        # return one of the two with uniform probability
        new_bt = np.random.choice(a=[parent_a, parent_b])

        return new_bt

    @staticmethod
    def generate(
        action_node_classes: list[type[ActionNode]],
        condition_node_classes: list[type[ConditionNode]],
        composite_node_classes: list[type[CompositeNode]],
        min_children: int = 3,
    ) -> "BehaviorTree":
        """Create a new behavior tree with at least min_children child nodes.
        #! Would be great to set a minimum depth instead of width.

        Args:
            action_node_classes list[type[ActionNode]]: list of available classes for action nodes,
            condition_node_classes list[type[ConditionNode]]: list of available classes for condition nodes,
            composite_node_classes list[type[CompositeNode]]: list of available classes for composite nodes,
            min_children (int, optional): minimum number of children for the root

        Returns:
            BehaviorTree: the newly instantiated behavior tree.
        """
        bt = BehaviorTree(
            action_node_classes, condition_node_classes, composite_node_classes
        )
        for _ in range(min_children):
            child_class: type[BehaviorNode] = np.random.choice(
                action_node_classes + condition_node_classes + composite_node_classes
            )
            if child_class in composite_node_classes:
                child = child_class.get_random_node(
                    action_node_classes, condition_node_classes, composite_node_classes
                )
            else:
                child = child_class.get_random_node()
            bt.root.insert_child(child, len(bt.root.children))
        return bt

    def tick(self, input: np.ndarray) -> tuple[BehaviorStates, np.ndarray]:
        return self.root.tick(input)

    def reset(self):
        """
        reset the memory of the tree
        """
        self.root.reset()

    def to_json(self, filename: str):
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
        while len(fifo) > 0:
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
            json.dump(all_nodes, outfile, indent=2)

    @staticmethod
    def from_json(
        filename: str,
        action_node_classes: list[type[ActionNode]],
        condition_node_classes: list[type[ConditionNode]],
        composite_node_classes: list[type[CompositeNode]],
    ) -> "BehaviorTree":
        """Creates a BT give a certain json which specifies the structure of
        the nodes. The format of the json must be the same as returned from
        method to_json(). There must be a node with index 0, which is the root.

        Args:
            filename (str): path to the file containing the json description.
            action_node_classes (list[type[ActionNode]]): list of available classes for action nodes,
            condition_node_classes (list[type[ConditionNode]]): list of available classes for condition nodes,
            composite_node_classes (list[type[CompositeNode]]): list of available classes for composite nodes,
        """
        name_to_class = {
            cl.__name__: cl
            for cl in (
                composite_node_classes + action_node_classes + condition_node_classes
            )
        }
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
        for composite_idx, children_idxs in relations.items():
            for child_idx in children_idxs:
                all_nodes[composite_idx].insert_child(all_nodes[str(child_idx)])
        # finally, create a new bt
        new_bt = BehaviorTree(
            action_node_classes, condition_node_classes, composite_node_classes
        )
        # and set the root
        new_bt.root = all_nodes["0"]
        return new_bt

    def is_regular(self) -> bool:
        """Checks if the tree is regular, namely if it does not contain two
        times the very same node.
        """
        tree_nodes: set[BehaviorNode] = set()
        nodes = [self.root]
        while len(nodes) > 0:
            node = nodes.pop()
            if hasattr(node, "children"):
                for child in node.children:
                    nodes.append(child)
            if node.id in tree_nodes:
                return False
            tree_nodes.add(node.id)
        return True

    def copy(self) -> "BehaviorTree":
        """Manual implementation of deepcopy."""
        copy = BehaviorTree(
            self.action_node_classes,
            self.condition_node_classes,
            self.composite_node_classes,
        )
        copy.root = self.root.copy()
        return copy

    def get_size(self) -> tuple[int, int]:
        return self.root.get_size()


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bt_lib.composite_nodes import composite_node_classes
    from derk.action_nodes import action_node_classes
    from derk.condition_nodes import condition_node_classes

    candidate_classes = (
        action_node_classes + condition_node_classes + composite_node_classes
    )
    name_to_class = {cl.__name__: cl for cl in (candidate_classes)}

    bt = BehaviorTree.generate(
        action_node_classes, condition_node_classes, composite_node_classes, 3
    )
    print(bt)
    # bt.to_json("try.json")
    # print("===============")
    # loaded = BehaviorTree.from_json("try.json",name_to_class)
    # print(loaded)
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
