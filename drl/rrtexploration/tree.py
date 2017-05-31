from anytree import Node, RenderTree, Walker
from node import TrajectoryNode
from trajectory import Trajectory
import numpy as np


class Tree(object):

    def __init__(self):
        self.nodes = {}
        self.root = TrajectoryNode('root', None, None)
        self.nodes[0] = self.root
        self.ind = 1
        self.parent = self.root
        self.w = Walker()

    def add_trajectory(self, trajectory):
        node = TrajectoryNode(str(self.ind), trajectory, self.parent)
        self.nodes[self.ind] = node
        self.ind += 1
        self.parent = node
        return node, self.ind-1

    def split_trajectory(self, node_ind, split_ind):
        node = self.nodes[node_ind]

        traj_1, traj_2 = node.trajectory.split(split_ind)

        self.parent = node.parent
        return_node, _ = self.add_trajectory(traj_1)
        self.add_trajectory(traj_2)

        for child in node.children:
            child.parent = self.parent

        node.parent = None
        self.nodes.pop(node_ind)

        return return_node

    def trajectory_to_node(self, node):
        walk = self.w.walk(self.root, node)
        states = []
        actions = []
        for node in walk[2]:
            cur_states = node.trajectory.get_states()
            cur_actions = node.trajectory.get_actions()
            for i in range(node.trajectory.size()):
                states.append(cur_states[i])
                actions.append(cur_actions[i])

        return np.array(states), np.array(actions)

    def print_tree(self):
        for pre, _, node in RenderTree(self.root):
            print("%s%s: %s" % (pre, node.name, node.trajectory))


def main():
    tree = Tree()

    for i in xrange(6):
        trajectory = Trajectory()
        for j in xrange(4):
            trajectory.add_node(np.random.rand(2), np.random.rand(1))
        tree.add_trajectory(trajectory)

    tree.split_trajectory(2, 3)


if __name__ == "__main__":
    main()
