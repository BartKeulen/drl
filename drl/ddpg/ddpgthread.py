import threading
import gym
import tensorflow as tf
import time

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, LinearDecay

env_name = 'HalfCheetah-v1'
save_results = False

options_algo = {
    'batch_norm': False,
    'l2_critic': 0.,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300]
}

options_agent = {
    'render_env': False,
    'num_episodes': 7500,
    'max_steps': 1000,
    'num_exp': 1,
    'print': False
}

options_noise = {
    'mu': 0.,
    'theta': 0.2,
    'sigma': 0.15,
    'start': 5000,
    'end': 7500
}

num_threads = 2


class DDPGThread(object):

    def __init__(self, thread_id):
        self.thread_id = thread_id

        env = gym.make(env_name)

        ddpg = DDPG(env=env,
                    options_in=options_algo)

        noise = OrnSteinUhlenbeckNoise(action_dim=env.action_space.shape[0])
        noise = LinearDecay(noise, options_in=options_noise)

        self.agent = RLAgent(env=env,
                            algo=ddpg,
                            exploration=noise,
                            options_in=options_agent,
                            save=save_results)

    def run(self, sess):
        print('Thread %d started.' % self.thread_id)
        self.agent.run_single_experiment(sess, self.thread_id)
        print('Thread %d finished.' % self.thread_id)


start = time.time()

threads = [DDPGThread(i) for i in range(num_threads)]

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

sess.run(tf.global_variables_initializer())


def train_func(thread_id):
    thread = threads[thread_id]

    thread.run(sess)

train_threads = [threading.Thread(target=train_func, args=(i,)) for i in range(num_threads)]

for t in train_threads:
    t.start()

for t in train_threads:
    t.join()

end = time.time()

print('Time elapsed: %.2f s' % (end - start))
