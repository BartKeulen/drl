from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.linalg as la


class DynamicsLR(object):

    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.fx = None
        self.fu = None
        self.fv = None
        self.dyn_covar = None

        self.models = []

    def fit(self, x, u):
        dX = x.shape[2]
        N, T, dU = u.shape

        self.fx = np.zeros((T, dX, dX))
        self.fu = np.zeros((T, dX, dU))
        self.fv = np.zeros((T, dX))
        self.dyn_covar = np.zeros((T, dX, dX))

        # Fit dynamics wih least squares regression.
        for t in range(T):
            X = np.c_[np.ones((x.shape[0], 1)), x[:, t, :], u[:, t, :]]
            Y = x[:, t + 1, :]

            beta = la.solve(X.T.dot(X) + np.eye(X.shape[1])*self.regularization, X.T.dot(Y))

            res = (Y - X.dot(beta))
            covar = res.T.dot(res)

            self.fv[t, :] = beta[1, :].T
            self.fx[t, :, :] = beta[1:1+dX, :].T
            self.fu[t, :, :] = beta[1+dX:, :].T
            self.dyn_covar[t, :, :] = covar.T

    def simulate(self, t, x, u, noise=None):
        x_new = self.fx[t, :, :].dot(x) + self.fu[t, :].dot(u)
        if noise is not None:
            x_new += self.dyn_covar[t, :, :].dot(noise)
        return x_new

    def derivatives(self, t, x, u):
        return self.fx[t, :, :], self.fu[t, :, :]

    def get(self, idx=None):
        if idx is None:
            return self.fx, self.fu, self.fv, self.dyn_covar
        return self.fx[idx, :, :], self.fu[idx, :], self.fv[idx, :], self.dyn_covar[idx, :]


class DynamicsFD(object):

    def __init__(self, dynamics, h=1e-6):
        self.dynamics = dynamics
        self.h = h

    def derivatives(self, t, x, u):
        fx, fu = finite_difference(self.dynamics, x, u)
        return fx, fu

    def simulate(self, t, x, u, noise=None):
        x_new = self.dynamics(x, u)
        return x_new


class Dynamics(object):

    def __init__(self, dynamics, fx, fu):
        self.dynamics = dynamics
        self.fx = fx
        self.fu = fu

    def derivatives(self, t, x, u):
        return self.fx, self.fu

    def simulate(self, t, x, u, noise=None):
        x_new = self.dynamics(x, u)
        return x_new, self.fx, self.fu


def finite_difference(func, x, u, h=1e-6):
    xu = np.concatenate((x, u), axis=0)
    n = x.shape[0]
    N = xu.shape[0]

    H = np.vstack((-h * np.eye(N), h * np.eye(N)))
    X = xu[None, :] + H
    Y = []
    for i in range(N*2):
        Y.append(func(X[i, :n], X[i, n:]))
    Y = np.array(Y)
    D = (Y[N:, :] - Y[:N, :])
    J = D / h / 2

    fx = J[:n, :].T
    fu = J[n:, :].T

    return fx, fu


if __name__ == "__main__":
    from drl.env.maze import Maze
    from drl.smartexploration.policy import LinearGaussianPolicy
    from drl.smartexploration.trajectory import Trajectory, TrajectoryList

    env = Maze.generate_maze(Maze.MEDIUM)
    dX, dU = env.observation_space.shape[0], env.action_space.shape[0]

    N = 5
    T = 1000
    dt = Maze.TIME_STEP

    ## Initial trajectory
    policy = LinearGaussianPolicy(T, dX, dU)
    policy.init_random()
    traj_list = TrajectoryList()

    for i in range(N):
        traj = Trajectory(T, dX, dU)
        x = env.reset()
        for t in range(T - 1):
            u = policy.act(t, x, np.random.randn(dU))

            traj.add(x, u)

            x, *_ = env.step(u)
        traj.add(x)
        # env.add_trajectory(np.copy(traj.get_X()), (255, 255, 255, 255))
        # env.render()
        traj_list.add(traj)

    dynamics = DynamicsLR()
    dynamics.fit(traj_list.get_X(), traj_list.get_U())

    x = env.reset()
    x_lr = x.copy()
    traj = Trajectory(T, dX, dU)
    traj_lr = Trajectory(T, dX, dU)
    for t in range(T-1):
        u = policy.act(t, x)
        traj.add(x, u)
        traj_lr.add(x_lr, u)
        x_lr = dynamics.simulate(t, x, u, np.random.randn(4))
        x, *_ = env.step(u)
        print(np.linalg.norm(x - x_lr))
    traj.add(x)
    traj_lr.add(x_lr)
    env.add_trajectory(np.copy(traj.get_X()), (255, 255, 255, 255))
    env.add_trajectory(np.copy(traj_lr.get_X()), (255, 0, 0, 255))
    env.render()

    while True:
        pass