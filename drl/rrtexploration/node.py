from anytree import Node, NodeMixin, RenderTree


class TrajectoryNode(NodeMixin):

    def __init__(self, name, trajectory, parent=None):
        super(TrajectoryNode, self).__init__()
        self.name = name
        self.trajectory = trajectory
        self.parent = parent
