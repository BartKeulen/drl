import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Lambda, merge, multiply
from keras.initializers import RandomUniform
import keras.backend as K
import numpy as np

w_init_low = RandomUniform(minval=-3e-4, maxval=3e-4)
w_init_high = RandomUniform(minval=-0.05, maxval=0.05)


class ActorNetwork(object):

    def __init__(self,
                 sess,
                 obs_dim,
                 action_dim,
                 action_bounds,
                 learning_rate,
                 tau,
                 hidden_nodes=[100, 100],
                 batch_norm=False):
        self.sess = sess
        self.learning_rate = learning_rate
        self.tau = tau
        self.hidden_nodes = hidden_nodes
        self.batch_norm = batch_norm

        K.set_session(self.sess)
        K.set_learning_phase(0)

        self.model, self.observations, self.weights = self._build_model(obs_dim, action_dim, action_bounds)
        self.params = self.model.trainable_weights + self.model.non_trainable_weights

        self.target_model, self.target_observations, self.target_weights = self._build_model(obs_dim, action_dim, action_bounds)
        self.target_params = self.target_model.trainable_weights + self.model.non_trainable_weights

        self.update_target_net_op = [self.target_params[i].assign(tf.multiply(self.params[i], self.tau) +
                                                                  tf.multiply(self.target_params[i], 1. - self.tau))
                                     for i in range(len(self.target_params))]

        self.action_gradients = tf.placeholder(tf.float32, [None, action_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradients)
        self.optim = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.params_grad, self.weights))

    def _build_model(self, obs_dim, action_dim, action_bounds):
        num_layers = len(self.hidden_nodes)

        x = Input(shape=[obs_dim], name='observations')

        h = x
        for i in range(num_layers):
            if self.batch_norm:
                h = BatchNormalization()(h)
            h = Dense(self.hidden_nodes[i],
                      activation='relu',
                      kernel_initializer=w_init_high,
                      bias_initializer='zeros',
                      name='h%s' % str(i))(h)

        mu = Dense(action_dim,
                   activation='tanh',
                   kernel_initializer=w_init_low,
                   bias_initializer='zeros',
                   name='mu')(h)

        mu = Lambda(lambda f: f*action_bounds)(mu)

        model = Model(inputs=x, outputs=mu)
        # model.summary()
        return model, x, model.trainable_weights

    def predict(self, observations):
        K.set_learning_phase(0)
        mu = self.model.predict(observations)
        return mu

    def predict_target(self, observations):
        K.set_learning_phase(0)
        mu = self.target_model.predict(observations)
        return mu

    def train(self, observations, action_gradients):
        K.set_learning_phase(1)
        self.sess.run(self.optim, {
            self.observations: observations,
            self.action_gradients: action_gradients
        })
        K.set_learning_phase(0)

    def init_target_net(self):
        self.target_model.set_weights(self.model.weights)

    def update_target_net(self):
        K.set_learning_phase(1)
        # self.sess.run(self.update_target_net_op)

        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = [weights[i] * self.tau + target_weights[i] * (1. - self.tau) for i in range(len(weights))]
        self.target_model.set_weights(new_weights)

        K.set_learning_phase(0)
