import tensorflow as tf
from replaybuffer import ReplayBuffer


class ReplayBufferTF(ReplayBuffer):

    def __init__(self, sess, obs_dim, buffer_size, random_seed=123):
        super(ReplayBufferTF, self).__init__(buffer_size, random_seed)
        self.sess = sess

        # Kernel density estimation op
        buffer_states = tf.placeholder(tf.float32, [None, obs_dim])

    # def calc_density(self, state, action):
