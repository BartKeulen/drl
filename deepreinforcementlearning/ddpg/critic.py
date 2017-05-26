import tensorflow as tf
from tensorflow.contrib.framework import get_variables
import tflearn


class CriticNetwork(object):

    def __init__(self,
                 sess,
                 obs_dim,
                 action_dim,
                 learning_rate,
                 hidden_nodes=[100, 100],
                 batch_norm=False,
                 scope="DDPG/critic"):
        self.sess = sess
        self.learning_rate = learning_rate
        num_layers = len(hidden_nodes)
        if not (num_layers >= 2):
            raise Exception("Critic network needs at least two hidden layers")

        with tf.variable_scope(scope):
            x = tf.placeholder(tf.float32, [None, obs_dim], name="observations")
            u = tf.placeholder(tf.float32, [None, action_dim], name="actions")

            h = x
            if batch_norm:
                # TODO: make trainable variable tensorflow placeholder
                h = tflearn.batch_normalization(h, trainable=True)
            h = tflearn.fully_connected(h, hidden_nodes[0], activation='relu')

            if batch_norm:
                # TODO: make trainable variable tensorflow placeholder
                h = tflearn.batch_normalization(h, trainable=True)

            t1 = tflearn.fully_connected(h, hidden_nodes[1])
            t2 = tflearn.fully_connected(u, hidden_nodes[1])

            h = tflearn.activation(tf.matmul(h, t1.W) + tf.matmul(u, t2.W) + t2.b, activation='relu')

            for i in range(2, num_layers):
                if batch_norm:
                    # TODO: make trainable variable tensorflow placeholder
                    h = tflearn.batch_normalization(h, trainable=True)
                h = tflearn.fully_connected(h, hidden_nodes[i], activation='relu')

            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            Q = tflearn.fully_connected(h, 1, weights_init=w_init)

            # Optimization algorithm
            self.y_target = tf.placeholder(tf.float32, [None, 1], "target_q")
            self.loss = tflearn.mean_square(self.y_target, Q)

            # Action gradient op
            self.action_gradients_op = tf.gradients(Q, u)

            # Make class variables
            self.variables = get_variables(scope)
            self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.observations, self.actions, self.Q = x, u, Q

    def predict(self, observations, actions):
        return self.sess.run(self.Q, {
            self.observations: observations,
            self.actions: actions
        })

    def train(self, observations, actions, y_target):
        _, Q, loss = self.sess.run([self.optim, self.Q, self.loss], {
            self.observations: observations,
            self.actions: actions,
            self.y_target: y_target
        })
        return Q, loss

    def action_gradients(self, observations, actions):
        return self.sess.run(self.action_gradients_op, {
            self.observations: observations,
            self.actions: actions
        })

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