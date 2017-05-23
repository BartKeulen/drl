import tensorflow as tf
from tensorflow.contrib.framework import get_variables
import tflearn


class ActorNetwork(object):

    def __init__(self,
                 sess,
                 obs_dim,
                 action_dim,
                 action_bounds,
                 learning_rate,
                 hidden_nodes=[100, 100],
                 scope="DDPG/actor"):
        self.sess = sess
        self.learning_rate = learning_rate
        num_layers = len(hidden_nodes)

        with tf.variable_scope(scope):
            x = tf.placeholder(tf.float32, [None, obs_dim], name="observations")

            # Create hidden layers
            h = x
            for i in xrange(num_layers):
                h = tflearn.fully_connected(h, hidden_nodes[i], activation='relu')
            w_init = tflearn.initializations.uniform(minval=-.003, maxval=0.003)
            outputs = tflearn.fully_connected(h, action_dim, activation='tanh', weights_init=w_init)
            mu = tf.multiply(outputs, action_bounds)

            # Get trainable variables
            variables = get_variables(scope)

            # Optimization algorithm
            action_gradients = tf.placeholder(tf.float32, [None, action_dim])
            gradients = tf.gradients(mu, variables, -action_gradients)
            self.optim = tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(gradients, variables))

            self.observations, self.mu, self.variables, self.action_gradients, self.gradients = \
                x, mu, variables, action_gradients, gradients

    def predict(self, observations):
        return self.sess.run(self.mu, {
            self.observations: observations
        })

    def train(self, observations, action_gradients):
        _, mu = self.sess.run([self.optim, self.mu], {
            self.observations: observations,
            self.action_gradients: action_gradients
        })

        return mu

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