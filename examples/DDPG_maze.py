import time
import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, LinearDecay
from drl.utilities import LinearScheduler
from drl.env import Pendulum

from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv

options_ddpg = {
    'batch_norm': False,
    'l2_critic': 0.,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300]
}

options_agent = {
    'render_env': True,
    'num_episodes': 250,
    'max_steps': 200,
    'num_exp': 5,
    'save_freq': 100,
    'record': False
}

start = time.time()

env = PointMazeEnv()

ddpg = DDPG(env=env,
            options_in=options_ddpg)

noise = OrnSteinUhlenbeckNoise(action_dim=env.action_space.shape[0], scale=env.action_space.high)
noise = LinearScheduler(noise, start=100, end=125)

agent = RLAgent(env=env,
                algo=ddpg,
                exploration=noise,
                options_in=options_agent)


with tf.Session() as sess:

    agent.run_experiment(sess)

    sess.close()


end = time.time()

print('Time elapsed: %.2f s' % (end - start))