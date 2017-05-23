import tensorflow as tf
import numpy as np
from replaybuffer import ReplayBuffer


class ReplayBufferTF(ReplayBuffer):

    def __init__(self, sess, obs_dim, obs_bounds, C, buffer_size, random_seed=123):
        super(ReplayBufferTF, self).__init__(buffer_size, random_seed)
        self.sess = sess

        self.obs_bounds = obs_bounds

        # Kernel density estimation op
        self.buffer_states = tf.placeholder(tf.float32, [None, obs_dim], name="buffer_states")
        self.state = tf.placeholder(tf.float32, [1, obs_dim], name="state")

        delta = tf.subtract(self.buffer_states, self.state)
        squared = tf.pow(delta, 2)
        scaled = tf.multiply(tf.div(squared, obs_bounds*2), C)
        summed = tf.reduce_sum(scaled, axis=1)
        self.density = tf.reduce_mean(tf.exp(-summed))

    def calc_density(self, state):
        return self.sess.run(self.density, feed_dict={
            self.buffer_states: np.array([_[0] for _ in self.buffer]),
            self.state: state
        })
