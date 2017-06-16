import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Lambda
from keras.initializers import RandomUniform, VarianceScaling
from keras.regularizers import l2
import keras.backend as K

# Variables for initializing neural networks
random_uniform_small = RandomUniform(minval=-3e-4, maxval=3e-4)
random_uniform_big = RandomUniform(minval=-0.05, maxval=0.05)


class ActorNetwork(object):
    """
    Actor network of the DDPG algorithm.
    """

    def __init__(self,
                 sess,
                 obs_dim,
                 action_dim,
                 action_bounds,
                 learning_rate,
                 tau,
                 l2_param,
                 hidden_nodes,
                 batch_norm):
        """
        Constructs 'ActorNetwork' object.

        :param sess: tensorflow session
        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :param action_bounds: upper limit of action space
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
        K.set_learning_phase(0)

        # Construct model for actor network
        self.model, self.observations, self.weights = self._build_model(obs_dim, action_dim, action_bounds)

        # Construct model for target actor network
        self.target_model, self.target_observations, self.target_weights = self._build_model(obs_dim, action_dim, action_bounds)

        # OP target network weight init
        self.init_target_net_op = [self.target_weights[i].assign(self.weights[i]) for i in range(len(self.target_weights))]

        # OP for soft target update of target network
        self.update_target_net_op = [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                                     for i in range(len(self.target_weights))]

        # OP for updating actor using policy gradient
        self.action_gradients = tf.placeholder(tf.float32, [None, action_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradients)
        self.optim = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.params_grad, self.weights))

    def _build_model(self, obs_dim, action_dim, action_bounds):
        """
        Builds the model for the actor network.

        Model consists of 'n' hidden layers with each 'm' hidden nodes.
            - 'n' is the length of hidden_nodes array
            - 'm' is corresponding entry value of hidden_nodes array

        If 'batch_norm = True' a batch normalization layer is added before each layer input.

        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :param action_bounds: upper limit of action space
        :return: model actor network, placeholder observation input, model weights
        """
        num_layers = len(self.hidden_nodes)
        x = Input(shape=[obs_dim], name='observations')

        h = x
        if self.batch_norm:
            h = BatchNormalization()(h)

        for i in range(num_layers):
            h = Dense(self.hidden_nodes[i],
                      activation='relu',
                      kernel_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform'),
                      bias_initializer='zeros',
                      kernel_regularizer=l2(self.l2_param),
                      name='h%s' % str(i))(h)
            # if self.batch_norm:
            #     h = BatchNormalization()(h)

        mu = Dense(action_dim,
                   activation='tanh',
                   kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                   bias_initializer='zeros',
                   kernel_regularizer=l2(self.l2_param),
                   name='mu')(h)

        mu = Lambda(lambda f: f*action_bounds)(mu)

        model = Model(inputs=x, outputs=mu)
        return model, x, model.trainable_weights

    def predict(self, observations):
        """
        Predicts the actions using actor network.

        :param observations: Tensor observations
        :return: Tensor actions
        """
        K.set_learning_phase(0)
        mu = self.model.predict(observations)
        return mu

    def predict_target(self, observations):
        """
        Predicts the actions using TARGET actor network.

        :param observations: Tensor observations
        :return: Tensor actions
        """
        K.set_learning_phase(0)
        mu = self.target_model.predict(observations)
        return mu

    def train(self, observations, action_gradients):
        """
        Trains the actor network using policy gradient as described in 'DDPG' class.

        :param observations: Tensor observations
        :param action_gradients: Tensor action gradients calculated by critic network
        """
        K.set_learning_phase(1)
        self.sess.run(self.optim, {
            self.observations: observations,
            self.action_gradients: action_gradients
        })
        K.set_learning_phase(0)

    def init_target_net(self):
        """
        Initializes the target actor network parameters to be equal to the actor network parameters.
        """
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        """
        Performs soft target update according to 'update_target_net_op'
        """
        K.set_learning_phase(1)
        self.sess.run(self.update_target_net_op)
        K.set_learning_phase(0)

    def print_summary(self):
        print('Summary actor network:')
        self.model.summary()

