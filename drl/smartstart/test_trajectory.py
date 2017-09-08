import numpy as np

from drl.env import Maze
from drl.replaybuffer import ReplayBufferKD
from drl.ilqg import ilqg, LearnedDynamics


max_steps = 150

buffer = ReplayBufferKD(10000)
env = Maze.generate_maze(Maze.MEDIUM)


obs_t = env.reset()
env.render()

for i in range(max_steps):
    action = np.random.randn(2)
    # action = np.array([-1, 0])

    obs_tp1, r, t, _ = env.step(action)

    buffer.add(obs_t, action, r, obs_tp1, t)

    obs_t = obs_tp1

    env.render()

trajectory = buffer.get_trajectory(buffer.parent)
env.add_trajectory(trajectory, (255, 255, 255, 255))

goal = obs_t
trajectory = buffer.get_trajectory(np.random.choice(buffer.parent_idxes))


env.render()

obs_t = env.reset()
for i in range(len(trajectory) - 1):
    obs_tr, action, *_ = trajectory[i]

    obs_tp1, r, t, _ = env.step(action)

    obs_t = obs_tp1

    env.render()
