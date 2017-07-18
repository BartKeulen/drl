import time

import gym
import tensorflow as tf
from drl.algorithms.ddpg import DDPG
from drl.algorithms.rlagent import RLAgent
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy, LinearDecay

env_name = 'Pendulum-v0'

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
    'num_exp': 5
}

options_noise = {
    'start': 100,
    'end': 125
}

start = time.time()

env = gym.make(env_name)
# env = Pendulum()

ddpg = DDPG(env=env,
            options_in=options_ddpg)

noise = OrnSteinUhlenbeckStrategy(action_dim=env.action_space.shape[0])
noise = LinearDecay(noise, options_in=options_noise)

agent = RLAgent(env=env,
                algo=ddpg,
                exploration_strategy=noise,
                options_in=options_agent)


with tf.Session() as sess:

    agent.run_experiment(sess)

    sess.close()


end = time.time()

print('Time elapsed: %.2f s' % (end - start))