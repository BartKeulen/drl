import tensorflow as tf
import numpy as np
from .replaybuffer import ReplayBuffer
import sys


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

    def calc_trans_min_density(self):
        min_density = sys.float_info.max
        min_transition = None

        for i in range(self.size()):
            transition = self.buffer[i]
            density = self.calc_density(transition[0].reshape([1, 2]))
            if density < min_density:
                min_density = density
                min_transition = transition

        return min_transition


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    with tf.Session() as sess:
        buffer = ReplayBufferTF(sess, 2, np.array([5., 2.]), 10., 500)

        for i in xrange(250):
            buffer.add((np.random.rand(2)-0.5)*[10, 4], None, None, None, None)

        for i in xrange(250):
            buffer.add(np.random.rand(2)*[2.5, 1.]+[2.5, 1.], None, None, None, None)

        s_buffer = np.array([_[0] for _ in buffer.buffer])

        fig1 = plt.figure()
        plt.scatter(s_buffer[:, 0], s_buffer[:, 1])

        resolution = 100
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-2, 2, resolution)

        xv, yv = np.meshgrid(x, y, sparse=False)
        values = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                values[i, j] = buffer.calc_density(np.array([[xv[i, j], yv[i, j]]]))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        contour = ax.contourf(xv, yv, values)

        min_transition = buffer.calc_trans_min_density()
        ax.plot(min_transition[0][0], min_transition[0][1], 'ro')

        plt.show()

if __name__ == "__main__":
    main()