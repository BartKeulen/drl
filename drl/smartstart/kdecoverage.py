import json
import os
import datetime

import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import ffmpy

from drl.env.maze import Maze
from drl.replaybuffer import ReplayBufferKD

path = '/home/bartkeulen/results/kdecoverage/'
render = True
render_step = False
save = True
timestamp = datetime.datetime.now().isoformat()


def run(smart_start, env, steps_reset, max_steps, exp_id):
    if smart_start:
        start_type = "smartstart"
    else:
        start_type = "randomstart"
    fp = os.path.join(path, timestamp, env.name, start_type, "%d" % exp_id)
    buffer = ReplayBufferKD(max_steps)

    obs_t = env.reset()
    if render:
        env.render()

    for step in tqdm(range(max_steps)):
        # Reset state to new initial state ever steps_reset
        if step == 0:
            obs_t = env.reset()
            env.render()

            fp_frames = os.path.join(fp, "frames")
            if not os.path.isdir(fp_frames):
                os.makedirs(fp_frames)

            fp_frames = os.path.join(fp_frames, "%06d.png" % step)
            env.save_frame(fp_frames)
        elif step % steps_reset == 0:
            if smart_start:
                samples, scores = buffer.kd_estimate(100)
                argmin_score = np.argmin(scores)
                x0 = samples[argmin_score]
            else:
                x0 = buffer.sample(1)[0][0]
                samples = None
            obs_t = env.reset(x0)
            if render:
                env.render(data=buffer.get_rgb_array(env), samples=samples)

                fp_frames = os.path.join(fp, "frames")
                if not os.path.isdir(fp_frames):
                    os.makedirs(fp_frames)

                fp_frames = os.path.join(fp_frames, "%06d.png" % step)
                env.save_frame(fp_frames)

        if render_step:
            env.render()

        action = np.random.randn(2)

        obs_tp1, r, t, _ = env.step(action)

        buffer.add(obs_t, action, r, obs_tp1, t)

        obs_t = obs_tp1

        # Save current observations in replay buffer
        if save and step % 100 == 0:
            fp_data = os.path.join(fp, "data")
            if not os.path.isdir(fp_data):
                os.makedirs(fp_data)

            fp_data = os.path.join(fp_data, "%06d.json" % step)
            json.dump(buffer.get_obs().tolist(), open(fp_data, "w"))

    if smart_start:
        samples, scores = buffer.kd_estimate(100)
        argmin_score = np.argmin(scores)
        x0 = samples[argmin_score]
    else:
        x0 = buffer.sample(1)[0][0]
        samples = None
    env.reset(x0)
    if render:
        env.render(data=buffer.get_rgb_array(env), samples=samples)

        fp_frames = os.path.join(fp, "frames")
        if not os.path.isdir(fp_frames):
            os.makedirs(fp_frames)

        fp_frames = os.path.join(fp_frames, "%06d.png" % max_steps)
        env.save_frame(fp_frames)

        fp_frames = os.path.join(fp, "frames")
        # Intialize ffmpy
        ff = ffmpy.FFmpeg(
            inputs={os.path.join(fp_frames, '*.png'): '-y -framerate 1 -pattern_type glob -s 720x720'},
            outputs={os.path.join(fp_frames, '%s-%s-%d.mp4' % (env.name, start_type, exp_id)): None}
        )
        # Send output of ffmpy to log.txt file in temporary record folder
        # if error check the log file
        ff.run(stdout=open(os.path.join(fp_frames, 'log.txt'), 'w'), stderr=open(os.path.join(fp_frames, 'tmp.txt'), 'w'))
        os.remove(os.path.join(fp_frames, 'log.txt'))
        os.remove(os.path.join(fp_frames, 'tmp.txt'))


if __name__ == "__main__":
    param_grid = {'maze': [Maze.SIMPLE, Maze.MEDIUM, Maze.COMPLEX],
                  'smartstart': [True, False],
                  'exp_id': range(5)}

    params = list(ParameterGrid(param_grid))

    for param in tqdm(params):
        if param['maze'] == Maze.SIMPLE:
            steps_reset = 1000
            max_steps = 10000
        elif param['maze'] == Maze.MEDIUM:
            steps_reset = 2250
            max_steps = 22500
        elif param['maze'] == Maze.COMPLEX:
            steps_reset = 4000
            max_steps = 40000
        else:
            raise Exception("Please choose from the available environments: [simple, medium]")

        env = Maze.generate_maze(param['maze'], False)
        run(param['smartstart'], env, steps_reset, max_steps, param['exp_id'])

