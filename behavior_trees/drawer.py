if __name__ == "__main__":

    import os
    import sys
    import networkx as nx
    from bt_lib.draw import BtDrawer
    from bt_lib.composite_nodes import composite_node_classes
    from bt_lib.behavior_tree import BehaviorTree
    from lunar_lander.action_nodes import action_node_classes
    from lunar_lander.condition_nodes import condition_node_classes

    bt = BehaviorTree.from_json(
        sys.argv[1],
        action_node_classes,
        condition_node_classes,
        composite_node_classes,
    )
    print(bt)
    a = BtDrawer(bt.root)
    a.draw()
