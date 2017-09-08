import numpy as np
from numpy.linalg import LinAlgError
import scipy.linalg as la

from drl.smartexploration.policy import LinearGaussianPolicy
from drl.smartexploration.trajectory import Trajectory, TrajectoryList
from drl.smartexploration.dynamics import DynamicsLR, DynamicsFD, Dynamics
from drl.utilities.numerical import function_derivatives
from drl.utilities.utilities import func_serializer


class TrajOptiLQG(object):

    def __init__(self,
                 max_itr=50,
                 lamb=1e-6,
                 lamb_min=1e-6,
                 lamb_max=1e4,
                 delta=2,
                 delta0=2.,
                 alpha=1.,
                 alpha_div=1.05,
                 alpha_min=0.05,
                 z0=1e-6):
        self.max_itr = max_itr
        self.lamb = lamb
        self.lamb_min = lamb_min
        self.lamb_max = lamb_max
        self.delta = delta
        self.delta0 = delta0
        self.alpha = alpha
        self.alpha_div = alpha_div
        self.alpha_min = alpha_min
        self.z0 = z0

    def optimize(self, dyn, cost_func, x, u):
        lamb = self.lamb
        delta = self.delta
        alpha = self.alpha

        print("Start trajectory optimization.")
        for itr in range(self.max_itr):
            # STEP 1 - Get cost and derivatives previous trajectory
            fx, fu, l, lx, lu, lxx, luu, lux = self.compute_derivatives(dyn, cost_func, x, u)
            print("Iter %d, initial cost: %.2f" % (itr, np.sum(l)))

            # STEP 2 - Backward pass
            backwardpass_done = False
            while not backwardpass_done:
                diverge, dV, Vx, Vxx, K, k, pol_covar = self.backward(fx, fu, lx, lu, lxx, luu, lux, lamb)

                if diverge:
                    delta = max(self.delta0, delta*self.delta0)
                    lamb = max(self.lamb_min, lamb*delta)
                    print("Backward pass diverged, increase lambda to: %.2f" % lamb)
                    if lamb > self.lamb_max:
                        print("Backward pass was unable to converge.")
                        break
                else:
                    delta = min(1/self.delta0, delta/self.delta0)
                    lamb = lamb * delta
                    if lamb < self.lamb_min:
                        lamb = 0.
                    backwardpass_done = True
                    print("Iter %d, Backward pass done. lambda: %.2f, dV: %.2f" % (itr, lamb, np.sum(dV)))

            # STEP 3 - Forward pass
            forwardpass_done = False
            if backwardpass_done:
                while not forwardpass_done:
                    x_new, u_new, l_new = self.forward(x, u, fx, fu, l, lx, lu, lxx, luu, lux, K, k, alpha)

                    dl = np.sum(l) - np.sum(l_new)

                    dJ = -np.sum(dV, axis=0)
                    dJ = alpha * (alpha * dJ[0] + dJ[1])
                    z = dl / dJ
                    if dl < 0 or z < self.z0:

                        alpha /= self.alpha_div
                        print("Iter %d, Forward pass diverged. Cost (old/new): (%.2f / %.2f). z: %.2f. Decrease alpha: %.2f" %
                              (itr, np.sum(l), np.sum(l_new), z, alpha))

                        # TODO: Check if integration diverged (what integration?)

                        if alpha < self.alpha_min:
                            delta = max(self.delta0, delta * self.delta0)
                            lamb = max(self.lamb_min, lamb * delta)
                            print("Iter %d, Forward pass diverged. Increase lambda to: %.2f" % (itr, lamb))
                            break
                    else:
                        print("Iter %d, Forward pass converged. Cost old / new: %.2f / %.2f" % (itr, np.sum(l), np.sum(l_new)))
                        forwardpass_done = True

            if forwardpass_done:
                x = x_new
                u = u_new
                l = l_new
                print("Iter %d, Trajectory optimization finished. Final cost: %.2f" % (itr, np.sum(l)))
                break
        else:
            print("FAILED: maximum number of iterations reached without convergence.")

        return x, u, l, K, k, pol_covar

    def compute_derivatives(self, dyn, cost_func, x, u):
        T, dim_x = x.shape
        dim_u = u.shape[1]

        fx = np.zeros((T, dim_x, dim_x))
        fu = np.zeros((T, dim_x, dim_u))

        l = np.zeros(T)
        lx = np.zeros((T, dim_x))
        lu = np.zeros((T, dim_u))
        lxx = np.zeros((T, dim_x, dim_x))
        luu = np.zeros((T, dim_u, dim_u))
        lux = np.zeros((T, dim_u, dim_x))

        for t in range(T - 1):
            fx[t, :], fu[t, :] = dyn.derivatives(t, x[t, :], u[t, :])

            l[t], lx[t, :], lu[t, :], lxx[t, :, :], luu[t, :, :], lux[t, :, :] = cost_func(x[t, :], u[t, :])

        l[T - 1], lx[T - 1, :], lu[T - 1, :], lxx[T - 1, :, :], luu[T - 1, :, :], lux[T - 1, :, :] = cost_func(
            x[T - 1, :], np.full_like(u[T - 2, :], np.nan))

        return fx, fu, l, lx, lu, lxx, luu, lux

    def forward(self, x, u, fx, fu, l, lx, lu, lxx, luu, lux, K, k, alpha):
        T, dim_x = x.shape
        dim_u = u.shape[1]

        x_new = np.zeros((T, dim_x))
        x_new[0, :] = x[0, :]
        u_new = np.zeros((T-1, dim_u))

        l_new = np.zeros(T)

        dX = np.zeros(dim_x)
        for t in range(T-1):
            dU = alpha * k[t, :] + K[t, :, :].dot(dX)
            dX = fx[t, :, :].dot(dX) + fu[t, :, :].dot(dU)

            u_new[t, :] = u[t, :] + dU
            x_new[t+1, :] = x[t+1, :] + dX

            l_new[t] = self._cost_approx(dX, dU, l[t], lx[t, :], lu[t, :], lxx[t, :, :], luu[t, :, :], lux[t, :, :])

        l_new[T - 1] = self._cost_approx(dX, np.zeros(dim_u), l[T - 1], lx[T - 1, :], lu[T - 1, :], lxx[T - 1, :, :], luu[T - 1, :, :], lux[T - 1, :, :])

        return x_new, u_new, l_new

    def _cost_approx(self, dX, dU, l, lx, lu, lxx, luu, lux):
        c = l + dX.T.dot(lx) + dU.T.dot(lu) + 0.5 * dX.T.dot(lxx).dot(dX) + 0.5 * dU.T.dot(luu).dot(dU) \
            + dU.T.dot(lux).dot(dX)
        return c

    def backward(self, fx, fu, lx, lu, lxx, luu, lux, lamb):
        T, dim_x = lx.shape
        dim_u = lu.shape[1]

        Qx = np.zeros((T-1, dim_x))
        Qu = np.zeros((T-1, dim_u))
        Qxx = np.zeros((T-1, dim_x, dim_x))
        Quu = np.zeros((T-1, dim_u, dim_u))
        Qux = np.zeros((T-1, dim_u, dim_x))

        dV = np.zeros((T-1, 2))
        Vx = np.zeros((T, dim_x))
        Vxx = np.zeros((T, dim_x, dim_x))

        K = np.zeros((T, dim_u, dim_x))
        k = np.zeros((T, dim_u))
        pol_covar = np.zeros((T, dim_u, dim_u))

        Vx[T-1, :] = lx[T-1, :]
        Vxx[T-1, :, :] = lxx[T-1, :, :]
        for t in range(T-2, -1, -1):
            Qx[t, :] = lx[t, :] + fx[t, :, :].T.dot(Vx[t+1, :])
            Qu[t, :] = lu[t, :] + fu[t, :, :].T.dot(Vx[t+1, :])
            Qxx[t, :, :] = lxx[t, :, :] + fx[t, :, :].T.dot(Vxx[t+1, :, :]).dot(fx[t, :, :])
            Quu[t, :, :] = luu[t, :, :] + fu[t, :, :].T.dot(Vxx[t+1, :, :]).dot(fu[t, :, :])
            Qux[t, :, :] = lux[t, :, :] + fu[t, :, :].T.dot(Vxx[t+1, :, :]).dot(fx[t, :, :])

            Quu_tilde = luu[t, :, :] + fu[t, :, :].T.dot(Vxx[t+1, :, :] + np.eye(dim_x) * lamb).dot(fu[t, :, :])
            Qux_tilde = lux[t, :, :] + fu[t, :, :].T.dot(Vxx[t+1, :, :] + np.eye(dim_x) * lamb).dot(fx[t, :, :])

            try:
                Quu_inv = la.inv(Quu_tilde)
            except LinAlgError as e:
                print("LinAlgError %s" % e)
                return True, dV, Vx, Vxx, K, k, pol_covar

            # Calculate control parameters
            K[t, :, :] = -Quu_inv.dot(Qux_tilde)
            k[t, :] = -Quu_inv.dot(Qu[t, :])
            pol_covar[t, :, :] = -Quu_inv

            # Update value function
            dV[t, 0] = 0.5 * k[t, :].T.dot(Quu[t, :, :]).dot(k[t, :])
            dV[t, 1] = k[t, :].T.dot(Qu[t, :])
            Vx[t, :] = Qx[t, :] + K[t, :, :].T.dot(Quu[t, :, :]).dot(k[t, :]) + K[t, :, :].T.dot(Qu[t, :]) \
                       + Qux[t, :, :].T.dot(k[t, :])
            Vxx[t, :, :] = Qxx[t, :, :] + K[t, :, :].T.dot(Quu[t, :, :]).dot(K[t, :, :]) \
                           + K[t, :, :].T.dot(Qux[t, :, :]) + Qux[t, :, :].T.dot(K[t, :, :])

        return False, dV, Vx, Vxx, K, k, pol_covar


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


def _cost_func(x_goal, x, u, wp, wu):
    x_goal = np.pad(x_goal[:2], (0, 2), 'constant')

    if np.isnan(u).any(): # Final cost
        l = wp * np.sum((x - x_goal)**2)
        lx = 2 * wp * (x - x_goal)
        lxx = 2 * wp * np.eye(x.shape[0])
        lu = np.zeros(u.shape)
        luu = np.zeros((u.shape[0], u.shape[0]))
        lux = np.zeros((u.shape[0], x.shape[0]))
    else: # Intermediate cost
        l = wu * np.sum(u*u)
        lx = np.zeros(x.shape)
        lu = 2 * wu * u
        lxx = np.zeros((x.shape[0], x.shape[0]))
        luu = 2 * wu * np.eye(u.shape[0])
        lux = np.zeros((u.shape[0], x.shape[0]))

    return l, lx, lu, lxx, luu, lux

if __name__ == "__main__":
    from drl.env.maze import Maze

    fp = "/home/bartkeulen/results/ilqg/init"

    env = Maze.generate_maze(Maze.EMPTY, goal=None)
    dim_x, dim_u = env.observation_space.shape[0], env.action_space.shape[0]

    T = 500
    wp = 1.
    wu = 1e-6

    # Initial trajectory
    policy = LinearGaussianPolicy(T, dim_x, dim_u)
    policy.init_random()
    init_traj = Trajectory(T, dim_x, dim_u)
    x = env.reset()
    for t in range(T - 1):
        u = policy.act(t, x, np.random.randn(dim_u))

        init_traj.add(x, u)

        x, *_ = env.step(u)
        env.render()
        env.save_frame(fp)
    init_traj.add(x)
    env.add_trajectory(np.copy(init_traj.get_X()), (255, 255, 255, 255))
    env.render()
    env.save_frame(fp)

    # Initialize goal
    x_goal = x[:2]
    print("x_goal: %s" % x_goal)
    env.set_initial_states(x_goal.reshape((1, 2)))
    env.render()

    # Set up cost function and dynamics
    cost_func = lambda x, u: _cost_func(x_goal, x, u, wp, wu)
    dyn = DynamicsFD(env.dynamics)
    # dyn = Dynamics(env.dynamics, env.A, env.B)

    alphas = [1.0, 0.95, 0.9, 0.5]
    for alpha in alphas:
        fp = "/home/bartkeulen/results/ilqg/alpha=%.2f" % alpha
        env.trajectories = []
        env.add_trajectory(np.copy(init_traj.get_X()), (255, 255, 255, 255))
        env.render()
        env.save_frame(fp)
        traj = init_traj.copy()

        ## Perform trajectory optimization
        traj_opt = TrajOptiLQG(alpha=alpha)
        for i in range(5):
            *_, l, K, k, _ = traj_opt.optimize(dyn, cost_func, traj.get_X(), traj.get_U())

            x, u = traj.get_X(), traj.get_U()

            # u = np.zeros(u_new.shape)
            traj = Trajectory(T, dim_x, dim_u)
            x_new = env.reset()
            l = 0.
            for t in range(T-1):
                u_new = u[t, :] + traj_opt.alpha * k[t, :] + K[t, :, :].dot(x_new - x[t, :])
                traj.add(x_new, u_new)
                l_new, *_ = _cost_func(x_goal, x_new, u_new, wp, wu)
                l += l_new
                x_new, *_ = env.step(u_new)
                env.render()
                env.save_frame(fp)
            traj.add(x_new)

            u_new = np.zeros(dim_u)
            u_new.fill(np.nan)
            l_new, *_ = _cost_func(x_goal, x_new, u_new, wp, wu)
            l += l_new
            print("x: %s, x_goal: %s, distance: %.2f" % (x_new[:2], x_goal, np.sum((x_new[:2] - x_goal[:2])**2)))
            print("Final cost: %.2f" % l)
            env.add_trajectory(np.copy(traj.get_X()), (255, 0, 0, 255))
            env.render()
            env.save_frame(fp)

    while True:
        pass
