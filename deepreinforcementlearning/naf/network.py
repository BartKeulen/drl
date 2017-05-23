import tensorflow as tf
from tensorflow.contrib.framework import get_variables
import tflearn

random_uniform_big = tf.random_uniform_initializer(-0.05, 0.05)
random_uniform_small = tf.random_uniform_initializer(-3e-4, 3e-4)


class NAFNetwork(object):

    def __init__(self,
                 sess,
                 obs_dim,
                 action_dim,
                 action_bounds,
                 learning_rate,
                 hidden_nodes=[100, 100],
                 use_batch_norm=True,
                 scope="NAF"
                 ):
        self.sess = sess
        self.learning_rate = learning_rate
        num_layers = len(hidden_nodes)

        with tf.variable_scope(scope):
            x = tf.placeholder(tf.float32, [None, obs_dim], name="observations")
            u = tf.placeholder(tf.float32, [None, action_dim], name="actions")
            phase = tf.placeholder(tf.bool, name="phase")

            # Create hidden layers
            h = x
            for i in xrange(num_layers):
                # if use_batch_norm:
                #     h = tflearn.batch_normalization(h, trainable=phase)
                h = tflearn.fully_connected(h, hidden_nodes[i], activation='relu')

            # Create V, mu and L networks
            V = tflearn.fully_connected(h, 1, activation='linear')
            mu = tflearn.fully_connected(h, action_dim, activation='tanh')
            mu = tf.multiply(mu, action_bounds)
            l = tflearn.fully_connected(h, action_dim, activation='linear')

            L = tf.exp(l)
            P = L*L
            A = -(u - mu)**2 * P
            Q = V + A

            # Optimization algorithm
            self.y_target = tf.placeholder(tf.float32, [None, 1], name="target_y")
            self.loss = tflearn.mean_square(self.y_target, Q)
            self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            # Make class variables
            self.is_train = phase
            self.variables = get_variables(scope)

            self.observations, self.actions, self.V, self.mu, self.P, self.A, self.Q = \
                x, u, V, mu, P, A, Q

    def predict_v(self, observations):
        return self.sess.run(self.V, {
            self.observations: observations,
            self.is_train: False
        })

    def predict_mu(self, observations):
        return self.sess.run(self.mu, {
            self.observations: observations,
            self.is_train: False
        })

    def train(self, observations, actions, y_target):
        _, q, v, a, mu, loss = self.sess.run([self.optim, self.Q, self.V, self.A, self.mu, self.loss], {
            self.observations: observations,
            self.actions: actions,
            self.y_target: y_target,
            self.is_train: True
        })
        return q, v, a, mu, loss

    def hard_copy_from(self, network):
        assert len(network.variables) == len(self.variables)

        for from_, to_ in zip(network.variables, self.variables):
            self.sess.run(to_.assign(from_))

    def make_soft_update_ops(self, network, tau):
        self.soft_update = {}
        for from_, to_ in zip(network.variables, self.variables):
            self.soft_update[to_.name] = to_.assign(from_ * tau + to_ * (1 - tau))

    def do_soft_update(self):
        for variable in self.variables:
            self.sess.run(self.soft_update[variable.name])
        return True
