import numpy as np


class Trajectory(object):

    def __init__(self, T, dX, dU):
        self.T = T
        self.dX = dX
        self.dU = dU
        self.X = np.zeros((T, dX))
        self.U = np.zeros((T-1, dU))
        self.count = 0

    def add(self, x, u=None):
        if len(x.shape) == 1:
            x = x[np.newaxis]
        self.X[self.count:self.count+x.shape[0], :] = x

        if u is not None:
            if len(u.shape) == 1:
                u = u[np.newaxis]
            assert x.shape[0] == u.shape[0]
            self.U[self.count:self.count+x.shape[0], :] = u
        self.count += x.shape[0]

    def get_X(self, idx=None):
        if idx is None:
            return self.X
        else:
            return self.X[idx, :]

    def get_U(self, idx=None):
        if idx is None:
            return self.U
        else:
            return self.U[idx, :]

    def copy(self):
        trajectory = Trajectory(self.T, self.dX, self.dU)
        trajectory.X = np.copy(self.X)
        trajectory.U = np.copy(self.U)
        trajectory.count = self.count
        return trajectory


class TrajectoryList(object):

    def __init__(self):
        self.T = None
        self.dX = None
        self.dU = None
        self.trajectories = []

    def add(self, trajectories):
        if type(trajectories) is Trajectory:
            trajectories = [trajectories]

        self._set_vars(trajectories)
        self.trajectories += trajectories

    def _set_vars(self, trajectories):
        if self.T is None and self.dX is None and self.dU is None:
            self.T = trajectories[0].T
            self.dX = trajectories[0].dX
            self.dU = trajectories[0].dU

        for trajectory in trajectories:
            assert trajectory.T == self.T
            assert trajectory.dX == self.dX
            assert trajectory.dU == self.dU

    def get_X(self, idx=None):
        return np.asarray([traj.get_X(idx) for traj in self.trajectories])

    def get_U(self, idx=None):
        return np.asarray([traj.get_U(idx) for traj in self.trajectories])

    def get_samples(self, idx=None):
        if idx is None:
            idx = range(len(self.trajectories))
        elif type(idx) == int:
            return self.trajectories[idx]
        return [self.trajectories[i] for i in idx]

    def len(self):
        return len(self.trajectories)

