import time

import gym
import tensorflow as tf
from drl.algorithms.ddpg import DDPG
from drl.env import GymEnv
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy
from drl.utilities.scheduler import LinearScheduler

env_name = 'Pendulum-v0'
num_experiments = 5

options_ddpg = {
    'batch_norm': False,
    'l2_critic': 0.01,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300]
}

options_agent = {
    'render_env': False,
    'num_episodes': 200,
    'max_steps': 200,
}

start = time.time()

env = GymEnv(env_name)

exploration_strategy = OrnSteinUhlenbeckStrategy(action_dim=env.action_space.shape[0])
exploration_decay = LinearScheduler(exploration_strategy, start=100, end=125)

agent = DDPG(env=env,
             l2_critic=0.01,
             batch_norm=False,
             render_env=True,
             num_episodes=10,
             max_steps=200)

with tf.Session() as sess:
    for run in range(num_experiments):
        agent.train(sess, run)

    sess.close()

end = time.time()

print('Time elapsed: %.2f s' % (end - start))