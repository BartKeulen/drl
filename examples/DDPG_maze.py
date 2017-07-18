import time

import tensorflow as tf
from drl.algorithms.ddpg import DDPG
from drl.algorithms.rlagent import RLAgent
from drl.env import SimpleMaze, MountainCar
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy, WhiteNoiseStrategy
from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
import gym


options_ddpg = {
    'batch_norm': False,
    'l2_critic': 0.,
    'lr_actor': 0.0001,
    'lr_critic': 0.001,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300],
    'buffer_size': 1000000,
    'prioritized_replay': False,
    'smart_start': False
}

options_agent = {
    'render_env': False,
    'num_episodes': 500,
    'max_steps': 500,
    'num_exp': 5,
    'save_freq': 50,
    'record': True,
    'num_tests': 1
}

options_exploration = {
    'sigma': 1.
}

start = time.time()

# env = SimpleMaze()
# env = PointMazeEnv()
env = MountainCar()

ddpg = DDPG(env=env,
            options_in=options_ddpg)

# exploration_strategy = OrnSteinUhlenbeckStrategy(action_dim=env.action_space.shape[0],
#                                                  scale=env.action_space.high)

exploration_strategy = WhiteNoiseStrategy(action_dim=env.action_space.shape[0],
                                          scale=env.action_space.high,
                                          options_in=options_exploration)

agent = RLAgent(env=env,
                algo=ddpg,
                exploration_strategy=exploration_strategy,
                options_in=options_agent)


with tf.Session() as sess:

    agent.run_experiment(sess)

    sess.close()


end = time.time()