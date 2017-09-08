import logging

import numpy as np

from drl.env import Maze
from drl.replaybuffer import ReplayBufferKD
from drl.trajopt_gps.guidedexploration import GuidedExploration
from drl.trajopt_gps.cost import cost_func

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
init_traj = buffer.get_traj_sample(parent_idx)

trajectory = buffer.get_trajectory(parent_idx)
env.add_trajectory(trajectory, (255, 255, 255, 255))

env.render()

gexpl = GuidedExploration(render=True)

wp = 1.
wu = 1.e-6
wv = 0.
c = lambda sample: cost_func(sample, init_traj.get_X()[-1, :2], wp, wv, wu)

gexpl.run(env, init_traj, c)