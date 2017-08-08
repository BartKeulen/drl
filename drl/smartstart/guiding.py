import numpy as np
from tqdm import tqdm

from drl.env import Maze
from drl.replaybuffer import ReplayBufferKD
from drl.ilqg import ilqg, LearnedDynamics

render = True
render_step = True


def run(smart_start, env, model, steps_reset, max_steps):
    buffer = ReplayBufferKD(max_steps)

    obs_t = env.reset()
    if render:
        env.render()

    for step in range(steps_reset):
        action = np.random.randn(2)

        obs_tp1, r, t, _ = env.step(action)

        buffer.add(obs_t, action, r, obs_tp1, t)

        obs_t = obs_tp1

    samples, scores = buffer.kd_estimate(100)
    argmin_score = np.argmin(scores)
    x0 = samples[argmin_score]

    obs_t = env.reset()
    env.render()

    for step in range(max_steps):
        


        action = np.random.randn(2)

        obs_tp1, r, t, _ = env.step(action)

        buffer.add(obs_t, action, r, obs_tp1, t)

        obs_t = obs_tp1


if __name__ == "__main__":
    steps_reset = 2250
    max_steps = 22500
    env = Maze.generate_maze(Maze.MEDIUM, False)

    N = 10
    Nf = 2
    model = LearnedDynamics(steps_reset, 10, env.observation_space.shape[0], env.action_space.shape[0], Nf)

