from anytree import NodeMixin


class TrajectoryNode(NodeMixin):

    def __init__(self, name, trajectory, parent=None):
        super(TrajectoryNode, self).__init__()
        self.name = name
        self.trajectory = trajectory
        self.parent = parent
