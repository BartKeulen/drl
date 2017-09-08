import numpy as np

from drl.smartexploration.policy import LinearGaussianPolicy
from drl.smartexploration.trajectory import Trajectory, TrajectoryList
from drl.smartexploration.dynamics import DynamicsLR, DynamicsFD
from drl.smartexploration.trajopt import TrajOptiLQG
from drl.env.maze import Maze


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


def cost_func(x_goal, x, u, wp, wu):
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


def simulate(env, policy, T, dim_x, dim_u, add_noise=False, render=False):
    traj = Trajectory(T, dim_x, dim_u)
    x = env.reset()
    noise = None
    for t in range(T - 1):
        if add_noise:
            noise = np.random.randn(dim_u)
        u = policy.act(t, x, noise)

        traj.add(x, u)

        x, *_ = env.step(u)

        if render:
            env.render()
    traj.add(x)
    return traj


def fit_dynamics(N, env, policy, T, dim_x, dim_u):
    traj_list = TrajectoryList()
    for i in range(N):
        traj = simulate(env, policy, T, dim_x, dim_u, add_noise=True)
        traj_list.add(traj)

    dynamics = DynamicsLR()
    dynamics.fit(traj_list.get_X(), traj_list.get_U())
    return dynamics


def main():
    env = Maze.generate_maze(Maze.EMPTY, goal=None)
    dim_x, dim_u = env.observation_space.shape[0], env.action_space.shape[0]

    N = 10
    T = 500
    alpha = 0.9

    # Initial trajectory
    policy = LinearGaussianPolicy(T, dim_x, dim_u)
    policy.init_random()
    traj = simulate(env, policy, T, dim_x, dim_u, render=True, add_noise=False)
    env.add_trajectory(np.copy(traj.get_X()), (255, 255, 255, 255))
    env.render()


    # Initialize goal
    x_goal = traj.get_X(-1)[:2]
    print("x_goal: %s" % x_goal)
    env.set_initial_states(x_goal.reshape((1, 2)))
    env.render()

    wp = 1.
    wu = 1e-6
    c = lambda x, u: cost_func(x_goal, x, u, wp, wu)

    # Perform trajectory optimization
    traj_opt = TrajOptiLQG(alpha=alpha)
    for i in range(10):
        # Execute N rollouts and fit dynamics
        dyn = fit_dynamics(N, env, policy, T, dim_x, dim_u)

        # Perform trajectory optimization
        x_new, u_new, l, K, k, pol_cov = traj_opt.optimize(dyn, c, traj.get_X(), traj.get_U())
        policy.fit(K, k, pol_cov, traj.get_X(), traj.get_U(), traj_opt.alpha)

        # Simulate policy
        traj = simulate(env, policy, T, dim_x, dim_u, render=True, add_noise=False)
        env.add_trajectory(np.copy(traj.get_X()), (255, 0, 0, 255))
        env.render()

        # print("x: %s, x_goal: %s, distance: %.2f" % (x[:2], x_goal, np.sum((x[:2] - x_goal[:2])**2)))
        # print("Final cost: %.2f" % l)

    while True:
        pass


if __name__ == "__main__":
    main()