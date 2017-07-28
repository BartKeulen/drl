import time

import tensorflow as tf
from drl.algorithms.ddpg import DDPG
from drl.env import Maze, MountainCar
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy, WhiteNoiseStrategy
import gym

num_experiments = 1

env = Maze.generate_maze(Maze.SIMPLE)


exploration_strategy = WhiteNoiseStrategy(action_dim=env.action_space.shape[0],
                                          scale=env.action_space.high)

ddpg = DDPG(env=env,
            exploration_strategy=exploration_strategy,
            num_episodes=100,
            max_steps=2000,
            smart_start=True,
            render_env=True)

with tf.Session() as sess:
    for i in range(num_experiments):
        print("\n\033[1mStarting experiment %d\033[0m" % i)
        ddpg.train(sess)