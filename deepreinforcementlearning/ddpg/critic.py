import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import keras.backend as K

w_init_low = RandomUniform(minval=-3e-4, maxval=3e-4)
w_init_high = RandomUniform(minval=-0.05, maxval=0.05)


class CriticNetwork(object):

    def __init__(self,
                 sess,
                 obs_dim,
                 action_dim,
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

        self.model, self.observations, self.actions = self._build_model(obs_dim, action_dim)
        self.target_model, self.target_observations, self.target_actions = self._build_model(obs_dim, action_dim)

        self.action_gradients_op = tf.gradients(self.model.output, self.actions)

    def _build_model(self, obs_dim, action_dim):
        num_layers = len(self.hidden_nodes)
        if not (num_layers >= 2):
            raise Exception("Critic network needs at least two hidden layers")

        x = Input(shape=[obs_dim], name='observations')
        u = Input(shape=[action_dim], name='actions')

        h = x
        if self.batch_norm:
            h = BatchNormalization()(h)
        h = Dense(self.hidden_nodes[0],
                  activation='relu',
                  kernel_initializer=w_init_high,
                  bias_initializer='zeros',
                  name='h0')(h)

        h = concatenate([h, u])
        if self.batch_norm:
            h = BatchNormalization()(h)

        h = Dense(self.hidden_nodes[1],
                  activation='relu',
                  kernel_initializer=w_init_high,
                  bias_initializer='zeros',
                  name='h1')(h)

        for i in range(2, num_layers):
            if self.batch_norm:
                h = BatchNormalization()(h)
            h = Dense(self.hidden_nodes[i],
                      activation='relu',
                      kernel_initializer=w_init_high,
                      bias_initializer='zeros',
                      name='h%s' % str(i))(h)

        Q = Dense(1,
                  activation='linear',
                  kernel_initializer=w_init_low,
                  bias_initializer='zeros',
                  name='Q')(h)

        model = Model(inputs=[x, u], outputs=Q)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, x, u

    def predict(self, observations, actions):
        K.set_learning_phase(0)
        q = self.model.predict([observations, actions])
        K.set_learning_phase(0)
        return q

    def predict_target(self, observations, actions):
        K.set_learning_phase(0)
        q = self.target_model.predict([observations, actions])
        K.set_learning_phase(0)
        return q

    def train(self, observations, actions, y_target):
        K.set_learning_phase(1)
        loss = self.model.train_on_batch([observations, actions], y_target)
        K.set_learning_phase(0)
        return loss

    def action_gradients(self, observations, actions):
        return self.sess.run(self.action_gradients_op, {
            self.observations: observations,
            self.actions: actions
        })

    def init_target_net(self):
        self.target_model.set_weights(self.model.weights)

    def update_target_net(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [target_weights[i] * (1 - self.tau) + weights[i] * self.tau for i in range(len(weights))]
        self.target_model.set_weights(target_weights)