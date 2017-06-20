import numpy as np
from drl.ilqg import ilqg, LearnedDynamics
from drl.env.arm import TwoLinkArm

env = TwoLinkArm(g=0., wp=10., wv=1., wu=0.001)
env.record_video()

N = 5 # number of future steps for iLQG
Nf = 2 # number of time-steps ahead and after current time-step for fitting linear model
num_episodes = 25
max_steps = 75

# Use full state access
full_state = True

# Initialize learned dynamics class
model = LearnedDynamics(max_steps, num_episodes, env.state_dim, env.action_dim, Nf)

x = env.reset(full_state=full_state)
x0 = x
goal = env.goal

# Initialize random control sequence
u = np.random.randn(max_steps, env.action_dim)

# Simulate system once
reward = 0.
for i_step in range(max_steps):
    env.render()

    x_new, r, t, _ = env.step(u[i_step, :], full_state=full_state)

    model.add(0, i_step, x, u[i_step, :], x_new)

    x = x_new
    reward += r
print('Iter %d, Steps %d, Reward: %.2f, Average reward: %.2f' % (0, i_step + 1, reward, reward / i_step))

env.record_video()
# Only use first N control inputs for iLQG estimator
u = u[:N, :]

for i_episode in range(1, num_episodes):
    # Fit models
    model.fit()

    x = env.reset(x0, goal, full_state=full_state)
    terminal = False
    i_step = 0
    reward = 0.

    for i_step in range(max_steps):
        env.render()

        model.set_cur_step(i_step)

        _, u, L, Vx, Vxx, cost = ilqg(model.dynamics_func, env.cost_func, x, u, {})

        # Take step
        x_new, r, t, _ = env.step(u[0, :], full_state=full_state)

        # Add to data matrices
        model.add(i_episode, i_step, x, u[0, :], x_new)

        u = np.concatenate((u[1:, :], np.random.randn(1, env.action_dim)))

        x = x_new
        reward += r
        i_step += 1

        if t:
            break

    print('Iter %d, Steps %d, Reward: %.2f, Average reward: %.2f' % (i_episode, i_step, reward, reward / i_step))