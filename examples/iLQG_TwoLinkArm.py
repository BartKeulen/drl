import numpy as np
from drl.ilqg import ilqg, LearnedDynamics
from drl.env.arm import TwoLinkArm

env = TwoLinkArm(g=0., wp=10., wv=1., wu=0.001)

N = 5 # number of future steps for iLQG
Nf = 2 # number of time-steps ahead and after current time-step for fitting linear model
num_episodes = 25
max_steps = 50

model = LearnedDynamics(max_steps, num_episodes, env.state_dim, env.action_dim, Nf)

env.reset()
x0 = env.q
x = x0.copy()
goal = env.goal

# Initialize random control sequence
u = np.random.randn(max_steps, env.action_dim)

# Simulate system once
i_step = 0
reward = 0.
done = False
for i_step in range(max_steps):
    x_new, _ = env.dynamics_func(x, u[i_step, :])

    reward += env.reward_func(x_new, u[i_step, :])
    done = env.terminal_func(x_new, u[i_step, :])

    model.add(0, i_step, x, u[i_step, :], x_new)

    x = x_new

    if done:
        break
print('Iter %d, Steps %d, Reward: %.2f, Average reward: %.2f' % (0, i_step + 1, reward, reward / i_step))

# Only use first N control inputs for iLQG estimator
u = u[:N, :]

for i_episode in range(1, num_episodes):
    # Fit models
    model.fit()

    env.reset(x0, goal)
    x = x0
    i_step = 0
    reward = 0.
    done = False

    for i_step in range(max_steps):
        _, u, L, Vx, Vxx, cost = ilqg(model.dynamics_func, env.cost_func, x, u, {})

        # Take step
        x_new, _ = env.dynamics_func(x, u[0, :])

        # reward += env.reward_func(x_new, u[i_step, :])
        # done = env.terminal_func(x_new, u[i_step, :])

        # Add to data matrices
        model.add(i_episode, i_step, x, u[0, :], x_new)

        u = np.concatenate((u[1:, :], np.random.randn(1, env.action_dim)))

        x = x_new
        i_step += 1

        if done:
            break

    print('Iter %d, Steps %d, Reward: %.2f, Average reward: %.2f' % (i_episode, i_step, reward, reward / i_step))

env.render(close=True)