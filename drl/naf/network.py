import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Lambda, merge
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import keras.backend as K

random_uniform_small = RandomUniform(minval=-3e-4, maxval=3e-4)
random_uniform_big = RandomUniform(minval=-0.05, maxval=0.05)


class NAFNetwork(object):

    def __init__(self,
                 sess,
                 obs_dim,
                 action_dim,
                 action_bounds,
                 learning_rate,
                 tau,
                 hidden_nodes=None,
                 batch_norm=True,
                 seperate_networks=False,
                 ):
        self.sess = sess
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.learning_rate = learning_rate
        self.tau = tau
        if hidden_nodes is None:
            self.hidden_nodes = [100, 100]
        else:
            self.hidden_nodes = hidden_nodes
        self.batch_norm = batch_norm
        self.seperate_networks = seperate_networks

        K.set_session(self.sess)
        K.set_learning_phase(1)

        self.model, self.observations, self.actions, self.V, self.mu = self._build_model()
        self.params = self.model.trainable_weights

        self.target_model, self.target_observations, self.target_actions, self.target_V, self.target_mu = self._build_model()
        self.target_params = self.target_model.trainable_weights

        self.update_target_net_op = [self.target_params[i].assign(tf.multiply(self.params[i], self.tau) +
                                                                  tf.multiply(self.target_params[i], 1. - self.tau))
                                     for i in range(len(self.target_params))]

        self.model.summary()
        K.set_learning_phase(0)

    def _build_model(self):
        x = Input(shape=[self.obs_dim], name='observations')
        u = Input(shape=[self.action_dim], name='actions')

        h = self._get_hidden_layers(x, 0)
        V = Dense(1,
                  activation='linear',
                  kernel_initializer=random_uniform_small,
                  bias_initializer='zeros',
                  name='V')(h)

        if self.seperate_networks:
            h = self._get_hidden_layers(x, 1)

        mu = Dense(self.action_dim,
                   activation='tanh',
                   kernel_initializer=random_uniform_small,
                   bias_initializer='zeros',
                   name='mu')(h)
        # mu = Lambda(self._scale_mu, output_shape=[self.action_dim], name="mu_scaled")(mu)

        if self.seperate_networks:
            h = self._get_hidden_layers(x, 2)

        l_net = Dense(self.action_dim * (self.action_dim + 1) / 2,
                      activation='linear',
                      kernel_initializer=random_uniform_small,
                      bias_initializer='zeros',
                      name='l_out')(h)

        L = Lambda(self._L, output_shape=[self.action_dim, self.action_dim], name="L")(l_net)
        P = Lambda(self._P, output_shape=[self.action_dim, self.action_dim], name="P")(L)
        A = Lambda(self._A, output_shape=[self.action_dim], name="A")([mu, P, u])
        Q = Lambda(self._Q, output_shape=[self.action_dim], name="Q")([V, A])

        fV = K.function([x], [V])
        fmu = K.function([x], [mu])
        self.fl_net = K.function([x], [l_net])
        self.fL = K.function([x], [L])
        self.fP = K.function([x], [P])

        model = Model(inputs=[x, u], outputs=Q)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, x, u, fV, fmu

    def _get_hidden_layers(self, x, net):
        num_layers = len(self.hidden_nodes)

        h = x
        for i in range(num_layers):
            if self.batch_norm:
                h = BatchNormalization(trainable=True)(h)
            h = Dense(self.hidden_nodes[i],
                      activation='relu',
                      kernel_initializer=random_uniform_big,
                      bias_initializer='zeros',
                      name='h%s%s' % (str(net), str(i)))(h)

        if self.batch_norm:
            h = BatchNormalization(trainable=True)(h)

        return h

    def _scale_mu(self, x):
        return tf.multiply(x, self.action_bounds)

    def _L(self, x):
        count = 0
        rows = []
        for i in range(self.action_dim):
            diag = tf.exp(tf.slice(x, (0, count), (-1, 1)))
            non_diag = tf.slice(x, (0, count + 1), (-1, i))
            count += i + 1
            row = tf.pad(tf.concat((non_diag, diag), axis=1), ((0, 0), (0, self.action_dim - i - 1)))
            rows.append(row)

        return tf.stack(rows, axis=1)

    @staticmethod
    def _P(x):
        return K.batch_dot(x, tf.transpose(x, (0, 2, 1)))

    @staticmethod
    def _A(t):
        mu, P, u = t
        d = K.expand_dims(u - mu, -1)

        return tf.reshape(-K.batch_dot(tf.transpose(d, (0, 2, 1)), K.batch_dot(P, d))/2., [-1, 1])

    @staticmethod
    def _Q(t):
        V, A = t
        return V + A

    def predict_q(self, observations, actions):
        q = self.model.predict([observations, actions])
        return q

    def predict_target_q(self, observations, actions):
        q = self.target_model.predict([observations, actions])
        return q

    def predict_v(self, observations):
        v = self.V([observations])
        return v[0]

    def predict_target_v(self, observations):
        v = self.target_V([observations])
        return v[0]

    def predict_mu(self, observations):
        mu = self.mu([observations])
        return mu[0]

    def predict_target_mu(self, observations):
        mu = self.target_mu([observations])
        return mu[0]

    def train(self, observations, actions, y_target):
        K.set_learning_phase(1)
        loss = self.model.train_on_batch([observations, actions], y_target)
        K.set_learning_phase(0)
        return loss

    def init_target_net(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)


def main(_):
    import numpy as np

    with tf.Session() as sess:
        network = NAFNetwork(sess, 2, 3, [1, 2, 3], 0.01, 0.01)

        sess.run(tf.global_variables_initializer())

        # x = tf.constant([1, 2], shape=[1, 2])
        # u = tf.constant([5, 6], shape=[1, 2])

        x = np.array([[1, 2], [3, 4]])
        u = np.array([[5, 6, 1], [7, 8, -3]])

        print("x: ", x)
        print("u: ", u)

        print("Q: ", network.model.predict([x, u]))
        print("")

        print("V: ", network.V([x])[0])
        print("V: ", network.V([x])[0].shape)
        print("mu: ", network.mu([x])[0])
        print("mu: ", network.mu([x])[0].shape)
        print("")

        print("l_net:\n", network.fl_net([x])[0])
        print("L    :\n", network.fL([x])[0])
        print("P    :\n", network.fP([x])[0])



if __name__ == "__main__":
    tf.app.run()