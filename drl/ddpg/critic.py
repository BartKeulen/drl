import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.initializers import RandomUniform, VarianceScaling
from keras.regularizers import l2
import keras.backend as K

# Variables for initializing neural networks
random_uniform_small = RandomUniform(minval=-3e-4, maxval=3e-4)
random_uniform_big = RandomUniform(minval=-0.05, maxval=0.05)


class CriticNetwork(object):
    """
    Critic network of the DDPG algorithm.
    """

    def __init__(self,
                 sess,
                 obs_dim,
                 action_dim,
                 learning_rate,
                 tau,
                 l2_param,
                 hidden_nodes,
                 batch_norm):
        """
        Constructs 'CriticNetwork' object

        :param sess: tensorflow session
        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :param learning_rate: learning rate
        :param tau: soft target update parameter
        :param l2_param: L2 regularization parameter
        :param hidden_nodes: array with each entry the number of hidden nodes in that layer.
                                Length of array is the number of hidden layers.
        :param batch_norm: True: use batch normalization otherwise False
        """
        self.sess = sess
        self.learning_rate = learning_rate
        self.tau = tau
        self.l2_param = l2_param
        self.hidden_nodes = hidden_nodes
        self.batch_norm = batch_norm

        # Set Keras session
        K.set_session(self.sess)
        K.set_learning_phase(1)

        # Construct model for critic network
        self.model, self.observations, self.actions, self.weights = self._build_model(obs_dim, action_dim)

        # Construct model for target critic network
        self.target_model, self.target_observations, self.target_actions, self.target_weights = self._build_model(obs_dim, action_dim)

        # OP target network weight init
        self.init_target_net_op = [self.target_weights[i].assign(self.weights[i]) for i in range(len(self.target_weights))]

        # OP for soft target update of target critic network
        self.update_target_net_op = [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                                     for i in range(len(self.target_weights))]

        # OP for calculating action gradient for policy gradient update of actor
        self.action_gradients_op = tf.gradients(self.model.output, self.actions)

        K.set_learning_phase(0)

    def _build_model(self, obs_dim, action_dim):
        """
        Builds the model for the critic network.

        Model consists of 'n' hidden layers with each 'm' hidden nodes.
            - 'n' is the length of hidden_nodes array
            - 'm' is corresponding entry value of hidden_nodes array

        If 'batch_norm = True' a batch normalization layer is added before each layer input.

        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :return: model critic network, placeholder observation input, placeholder action input, model weights
        """
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
                  kernel_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform'),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(self.l2_param),
                  name='h0')(h)

        h = concatenate([h, u])

        for i in range(1, num_layers):
            h = Dense(self.hidden_nodes[i],
                      activation='relu',
                      kernel_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform'),
                      bias_initializer='zeros',
                      kernel_regularizer=l2(self.l2_param),
                      name='h%s' % str(i))(h)

        Q = Dense(1,
                  activation='linear',
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(self.l2_param),
                  name='Q')(h)

        model = Model(inputs=[x, u], outputs=Q)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, x, u, model.trainable_weights

    def predict(self, observations, actions):
        """
        Predicts the Q-values using critic network.

        :param observations: Tensor observations
        :param actions: Tensor actions
        :return: Tensor Q-values
        """
        K.set_learning_phase(0)
        q = self.model.predict([observations, actions])
        K.set_learning_phase(0)
        return q

    def predict_target(self, observations, actions):
        """
        Predicts the Q-values using TARGET critic network.

        :param observations: Tensor observations
        :param actions: Tensor actions
        :return: Tensor Q-values
        """
        K.set_learning_phase(0)
        q = self.target_model.predict([observations, actions])
        K.set_learning_phase(0)
        return q

    def train(self, observations, actions, y_target):
        """
        Trains the critic network by minimizing loss as described in 'DDPG' class.

        :param observations: Tensor observations
        :param actions: Tensor actions
        :param y_target: Tensor targets (y(i))
        :return: loss
        """
        K.set_learning_phase(1)
        loss = self.model.train_on_batch([observations, actions], y_target)
        K.set_learning_phase(0)
        return loss

    def action_gradients(self, observations, actions):
        """
        Function for calculating 'action_gradients' using 'action_gradients_op'

        :param observations: Tensor observations
        :param actions: Tensor actions
        :return: Tensor action_gradients
        """
        return self.sess.run(self.action_gradients_op, {
            self.observations: observations,
            self.actions: actions
        })

    def init_target_net(self):
        """
        Initializes the target critic network parameters to be equal to the critic network parameters.
        """
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        """
        Performs soft target update according to 'update_target_net_op'
        """
        self.sess.run(self.update_target_net_op)

    def print_summary(self):
        print('Summary critic network:')
        self.model.summary()