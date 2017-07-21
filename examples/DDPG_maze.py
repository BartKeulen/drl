import time

import tensorflow as tf
from drl.algorithms.ddpg import DDPG
from drl.env import SimpleMaze, MountainCar
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy, WhiteNoiseStrategy
import gym

num_experiments = 1

# env = SimpleMaze()
# env = PointMazeEnv()
env = MountainCar()


exploration_strategy = WhiteNoiseStrategy(action_dim=env.action_space.shape[0],
                                          scale=env.action_space.high)

ddpg = DDPG(env=env,
            exploration_strategy=exploration_strategy,
            num_episodes=500,
            max_steps=500,
            save_freq=50)

with tf.Session() as sess:
    for i in range(num_experiments):
        print("\n\033[1mStarting experiment %d\033[0m" % i)
        ddpg.train(sess)