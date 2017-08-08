import numpy as np

from drl.env import Maze
from drl.replaybuffer import ReplayBufferKD
from drl.ilqg import ilqg, LearnedDynamics


env = Maze.generate_maze(Maze.MEDIUM, [2, 10])

N = 10
Nf = 2
num_episodes = 25
max_steps = 500

model = LearnedDynamics(max_steps, num_episodes, env.observation_space.shape[0], env.action_space.shape[0], Nf)

x = env.reset()
x0 = x
goal = env.get_goal()
env.render()

u = np.random.randn(max_steps, env.action_space.shape[0])

reward = 0.
i_step = 0
for i_step in range(max_steps):
    env.render()

    x_new, r, t, _ = env.step(u[i_step, :])
    model.add(0, i_step, x, u[i_step, :], x_new)
    x = x_new
    reward += r

print('Iter %d, Steps %d, Reward: %.2f, Average reward: %.2f' % (0, i_step + 1, reward, reward / i_step))

# Only use first N control inputs for iLQG estimator
u = u[:N, :]


def cost_func(x_hat, u_hat):
    final = np.isnan(u_hat)
    u_hat[final] = 0

    # cu = 0.001 * np.linalg.norm(u_hat)
    cu = 0.
    if final.any():
        cp = 10*np.linalg.norm(goal[:2] - x_hat[:2])
        # cv = np.linalg.norm(x_hat[2:])
    else:
        cp = 0
        # cv = 0
    c = cu + cp
    return c, None, None, None, None, None


for i_episode in range(1, num_episodes):
    # Fit models
    model.fit()

    x = env.reset()
    terminal = False
    reward = 0.
    cost = 0.

    for i_step in range(max_steps):
        env.render()

        model.set_cur_step(i_step)

        options_in = {
            'dyn_first_der': False,
            'cost_first_der': False,
            'cost_sec_der': False
        }
        _, u, L, Vx, Vxx, c = ilqg(env.dynamics_func, cost_func, x, u, options_in=options_in)

        # Take step
        x_new, r, t, _ = env.step(u[0, :])

        # Add to data matrices
        model.add(i_episode, i_step, x, u[0, :], x_new)

        u = np.concatenate((u[1:, :], np.random.randn(1, env.action_space.shape[0])))

        x = x_new
        cost += np.sum(c)
        reward += r
        i_step += 1

        if t:
            break

    print('Iter %d, Steps %d, Reward: %.2f, Average reward: %.2f, Cost: %.2f' % (i_episode, i_step, reward, reward / i_step, cost))