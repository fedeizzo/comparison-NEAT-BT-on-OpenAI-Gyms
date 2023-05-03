import matplotlib.pyplot as plt
import networkx as nx
from bt_lib.behavior_tree import BehaviorTree


class BtDrawer:
    def __init__(self, root):
        self._root = root
        self._max_level = 1
        self._final_nodes = []
        self._bt_graph = nx.Graph()

    def pre_draw(self):
        final_edges = []
        final_nodes = []
        nodes = [self._root]

        while nodes:  # children != [] or len(children) != 0
            new_nodes = []
            for node in nodes:
                if hasattr(node, "children"):
                    children = node.children
                else:
                    children = []
                # save a node and its edges
                final_edges.append((node, children))
                final_nodes.append((node, self._max_level))

                for child in children:
                    final_nodes.append((child, self._max_level + 1))
                    if child:
                        new_nodes.append(child)

            # update nodes for recursive iteration
            nodes = new_nodes
            self._max_level += 1

        # removing duplicates
        for node in final_nodes:
            if node not in self._final_nodes:
                self._final_nodes.append(node)

        # list comprehensionwith with double iterations
        self._bt_graph.add_edges_from(
            [(node, child) for (node, children) in final_edges for child in children]
        )

    @staticmethod
    def get_hierarchy_pos(
        G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, leaf_vs_root_factor=0.5
    ):
        """
        param G: the graph (must be a tree)
        root: the root node of the tree
        width: horizontal space allocated for this branch - avoids overlap with other branches
        vert_gap: gap between levels of hierarchy
        vert_loc: vertical location of root
        leaf_vs_root_factor:
        x_center: horizontal location of root
        """

        if not nx.is_tree(G):
            raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

        def _hierarchy_pos(
            G,
            root,
            leftmost,
            width,
            leaf_dx=0.2,
            vert_gap=0.2,
            vert_loc=0,
            x_center=0.5,
            root_pos=None,
            leaf_pos=None,
            parent=None,
        ):

            if root_pos is None:
                root_pos = {root: (x_center, vert_loc)}
            else:
                root_pos[root] = (x_center, vert_loc)
            if leaf_pos is None:
                leaf_pos = {}
            children = list(G.neighbors(root))
            leaf_count = 0
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                root_dx = width / len(children)
                next_x = x_center - width / 2 - root_dx / 2
                for child in children:
                    next_x += root_dx
                    root_pos, leaf_pos, new_leaves = _hierarchy_pos(
                        G,
                        child,
                        leftmost + leaf_count * leaf_dx,
                        width=root_dx,
                        leaf_dx=leaf_dx,
                        vert_gap=vert_gap,
                        vert_loc=vert_loc - vert_gap,
                        x_center=next_x,
                        root_pos=root_pos,
                        leaf_pos=leaf_pos,
                        parent=root,
                    )
                    leaf_count += new_leaves

                leftmost_child = min(
                    (x for x, y in [leaf_pos[child] for child in children])
                )
                rightmost_child = max(
                    (x for x, y in [leaf_pos[child] for child in children])
                )
                leaf_pos[root] = ((leftmost_child + rightmost_child) / 2, vert_loc)
            else:
                leaf_count = 1
                leaf_pos[root] = (leftmost, vert_loc)
            return root_pos, leaf_pos, leaf_count

        x_center = width / 2.0
        if isinstance(G, nx.DiGraph):
            leaf_count = len(
                [node for node in nx.descendants(G, root) if G.out_degree(node) == 0]
            )
        elif isinstance(G, nx.Graph):
            leaf_count = len(
                [
                    node
                    for node in nx.node_connected_component(G, root)
                    if G.degree(node) == 1 and node != root
                ]
            )
        root_pos, leaf_pos, leaf_count = _hierarchy_pos(
            G,
            root,
            0,
            width,
            leaf_dx=width * 1.0 / leaf_count,
            vert_gap=vert_gap,
            vert_loc=vert_loc,
            x_center=x_center,
        )
        pos = {}
        for node in root_pos:
            pos[node] = (
                leaf_vs_root_factor * leaf_pos[node][0]
                + (1 - leaf_vs_root_factor) * root_pos[node][0],
                leaf_pos[node][1],
            )
        x_max = max(x for x, y in pos.values())
        for node in pos:
            pos[node] = (pos[node][0] * width / x_max, pos[node][1])
        return pos

    def draw(self):
        self.pre_draw()
        # x,y are the coordinates/size of the drawing
        y = self._max_level
        x = int(0.5 + y / 2)
        root = self._final_nodes[0][0]
        del self._final_nodes[0]
        # import pdb; pdb.set_trace()
        # i need to set an id for each node
        # and create a custom mapping for each shape
        # then get the label
        # finally set a custom color for each node according to the tick
        positions = {root: (x, y)}  # all positions where to draw
        node_shapes = ["s"]  # all shapes
        labels = {root: type(root).__name__}  # all labels
        if hasattr(root, "parameters"):
            for key, label in root.parameters.items():
                labels[root] = labels[root] + " \n" + "{} = {:.2f}".format(key, label)
        for index, (node, level) in enumerate(self._final_nodes):
            node_shapes.append("s")
            labels[node] = type(node).__name__  # later change name by ID
            if hasattr(node, "parameters"):
                for key, label in node.parameters.items():
                    if type(label) == float:
                        labels[node] = (
                            labels[node] + " \n" + "{} = {:.2f}".format(key, label)
                        )
                    elif type(label) == str:
                        labels[node] = (
                            labels[node] + " \n" + "{} = {}".format(key, label)
                        )
        # gets the positions where to draw each node
        positions = self.get_hierarchy_pos(self._bt_graph, root)
        plt.figure("Your Behavior Tree")

        for index, node in enumerate(self._bt_graph.nodes()):  # gets each shape
            self._bt_graph.nodes[node]["shape"] = node_shapes[index]

        # for (key,label) in labels.items():
        #     if len(label) > 10:
        #         labels[key] = labels[key][:15] + "\n" + labels[key][15:]
        # import pdb; pdb.set_trace()
        nx.draw_networkx_edges(self._bt_graph, positions)  # draw edges
        nx.draw_networkx_labels(
            self._bt_graph, positions, labels, verticalalignment="center_baseline",font_size=8
        )  # draw node labels

        # Draw the nodes for each shape with the shape specified
        for shape in set(node_shapes):
            node_list = [
                node
                for node in self._bt_graph.nodes()
                if self._bt_graph.nodes[node]["shape"] == shape
            ]
            node_sizes = [1000] * len(node_list)
            node_colors = ["#FFFFFF"] * len(node_list)
            edge_colors = ["#000000"] * len(self._bt_graph.edges())
            nx.draw_networkx_nodes(
                self._bt_graph,
                positions,
                nodelist=node_list,
                node_color=node_colors,
                node_size=node_sizes,
                node_shape=shape,
                edgecolors=edge_colors,
            )

        # plt.axis('off') # this removes border
        plt.tight_layout()  # this is to have less wasted space
        plt.show()  # display


if __name__ == "__main__":

    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bt_lib.composite_nodes import composite_node_classes
    from derk.action_nodes import action_node_classes
    from derk.condition_nodes import condition_node_classes

    bt = BehaviorTree.from_json(
        "./behavior_trees/derk/saved_bts/dummy.json",
        action_node_classes,
        condition_node_classes,
        composite_node_classes,
    )
    print(bt)
    a = BtDrawer(bt.root)
    a.draw()
