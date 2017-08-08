import numpy as np

from drl.env import Maze
from drl.replaybuffer import ReplayBufferKD
from drl.ilqg import ilqg, LearnedDynamics


max_steps = 150

buffer = ReplayBufferKD(10000)
env = Maze.generate_maze(Maze.MEDIUM)


obs_t = env.reset()
env.render()

parent_idx = -1
for i in range(max_steps):
    action = np.random.randn(2)
    # action = np.array([-1, 0])

    obs_tp1, r, t, _ = env.step(action)

    parent_idx = buffer.add(obs_t, action, r, obs_tp1, t, parent_idx)

    obs_t = obs_tp1

    # env.render()

goal = obs_t
trajectory = buffer.get_trajectory(parent_idx)

env.add_trajectory(trajectory, (255, 255, 255, 255))

env.render()

N = 5
Nf = 2
model = LearnedDynamics(len(trajectory) - 1, 10, env.observation_space.shape[0], env.action_space.shape[0], Nf)

x = np.zeros((len(trajectory) - 1, env.observation_space.shape[0]))
U = np.zeros((len(trajectory) - 1, env.action_space.shape[0]))
for i in range(len(trajectory) - 1):
    node_t = trajectory[i]
    node_tp1 = trajectory[i+1]
    x[i, :] = node_t[0]
    U[i, :] = node_t[1]
    model.add(0, i, x[i, :], U[i, :], node_tp1[0])

u = U[:N, :]


def cost_func(x_hat, u_hat, goal):
    final = np.isnan(u_hat)
    u_hat[final] = 0

    # cu = 1e-6 * np.linalg.norm(u_hat)
    cu = 0.
    if final.any():
        cp = 10*np.linalg.norm(goal - x_hat)
    else:
        cp = 0
    c = cu + cp
    return c, None, None, None, None, None


for i_episode in range(1, 10):
    model.fit()

    obs_t = env.reset()
    done = False
    parent_idx = -1

    cost = 0.
    reward = 0.
    for i_step in range(max_steps - 1):
        if i_step % 10 == 0:
            env.render()

        model.set_cur_step(i_step)

        options_in = {
            'dyn_first_der': False,
            'cost_first_der': False,
            'cost_sec_der': False
        }

        # goal =
        cost_func_lmd = lambda x, u: cost_func(x, u, goal)
        x, u, L, Vx, Vxx, c = ilqg(env.dynamics_func, cost_func_lmd, obs_t, u, options_in)

        obs_tp1, r, t, _ = env.step(u[0, :])

        parent_idx = buffer.add(obs_t, u[0, :], r, obs_tp1, done, parent_idx)
        model.add(i_episode, i_step, obs_t, u[0, :], obs_tp1)

        if i_step + 1 >= U.shape[0]:
            u = u[1:, :]
        else:
            u = np.concatenate((u[1:, :], U[np.newaxis, i_step + 1, :]))

        obs_t = obs_tp1
        cost += np.sum(c)
        reward += r

    print("Episode %d, Reward: %.2f, cost: %.2f" % (i_episode, reward, cost))

    trajectory = buffer.get_trajectory(parent_idx)

    env.add_trajectory(trajectory, (255, 0, 0, 255))
    env.render()
