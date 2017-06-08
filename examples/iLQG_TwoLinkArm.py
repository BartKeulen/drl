import numpy as np
from drl.ilqg import ilqg
from drl.env.arm import TwoLinkArm

env = TwoLinkArm(g=0.)

dyn = lambda x, u: env.dynamics_func(x, u)[0]
cst = lambda x, u: env.cost_func(x, u)
N = 5 # number of future steps for iLQG
num_episodes = 10
max_steps = 100

for i_episode in range(num_episodes):
    x = env.reset()
    u = np.random.randn(N, env.action_dim) * 15.
    terminal = False
    i_step = 0
    reward = 0.

    while not terminal and i_step < max_steps:
        env.render()

        x, u, L, Vx, Vxx, cost = ilqg(dyn, cst, x, u, {})

        x, r, t, _ = env.step(u[0, :])

        reward += r
        i_step += 1

        if t:
            break

    print('Iter %d, Steps %d, Reward: %s' % (i_episode, i_step, reward))

env.render(close=True)