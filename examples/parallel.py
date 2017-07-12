from threading import Thread
import time
import os
import sys

import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise
from drl.env import Pendulum

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import conf_cheetah as conf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1/conf.options_agent['num_exp'], allow_growth=True)

agents = []
for i in range(conf.options_agent['num_exp']):
    graph = tf.Graph()

    with graph.as_default():
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        if hasattr(conf, 'gym') and conf.gym:
            env = gym.make(conf.env_name)
        else:
            env = Pendulum()

        ddpg = DDPG(env=env,
                    options_in=conf.options_algo)

        noise = OrnSteinUhlenbeckNoise(action_dim=env.action_space.shape[0])

        agent = RLAgent(env=env,
                        algo=ddpg,
                        exploration=noise,
                        options_in=conf.options_agent)

        sess.run(tf.global_variables_initializer())

        agents.append((graph, sess, agent))


def train_func(thread_id):
    print('Thread %d started.' % thread_id)
    graph, sess, agent = agents[thread_id]
    with graph.as_default():
        agent.train(sess, thread_id, parallel=True)
    print('Thread %d finished.' % thread_id)


threads = [Thread(target=train_func, args=(i,)) for i in range(conf.options_agent['num_exp'])]

for t in threads:
    t.start()

for t in threads:
    t.join()

for graph, sess, agent in agents:
    with graph.as_default():
        sess.close()