import tensorflow as tf

from drl.utilities import fc_layer, bn_layer, TFNetwork


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
        :param hidden_nodes: array with each entry the number of hidden nodes in that layer.
                                Length of array is the number of hidden layers.
        :param batch_norm: True: use batch normalization otherwise False
        """
        self.sess = sess
        self.learning_rate = learning_rate
        self.tau = tau
        self.hidden_nodes = hidden_nodes
        self.batch_norm = batch_norm

        # Boolean saying for phase of system, training or test
        self.training_phase = tf.placeholder(dtype=tf.bool, name='phase')

        # Construct model for actor network
        self.output, self.observations, self.network = self._build_model('actor', obs_dim, action_dim, action_bounds)

        # Construct model for target actor network
        self.target_output, self.target_observations, self.target_network = self._build_model('target_actor', obs_dim, action_dim, action_bounds)

        # Set weight variables
        self.weights = self.network.get_weights()
        self.target_weights = self.target_network.get_weights()

        # OP target network weight init
        self.init_target_net_op = [self.target_weights[i].assign(self.weights[i]) for i in range(len(self.target_weights))]

        # OP for soft target update of target network
        self.update_target_net_op = [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                                     for i in range(len(self.weights))]

        # OP for updating actor using policy gradient
        self.action_gradients = tf.placeholder(tf.float32, [None, action_dim])
        self.params_grad = tf.gradients(self.output, self.weights, -self.action_gradients)
        self.optim = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.params_grad, self.weights))

    def _build_model(self, name, obs_dim, action_dim, action_bounds):
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
        with tf.variable_scope(name):
            network = TFNetwork(name)
            num_layers = len(self.hidden_nodes)

            x = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim], name='observation')
            network.add_layer(x)
            h = x

            # Set layer_func to Fully-Connected or Batch-Normalization layer
            layer_func = fc_layer
            if self.batch_norm:
                layer_func = bn_layer

            # Hidden layers
            for i in range(num_layers):
                h, h_weights = layer_func(h, self.hidden_nodes[i], tf.nn.relu, i=i, phase=self.training_phase)
                network.add_layer(h, h_weights)

            # Output layer
            n_in = h.get_shape().as_list()[1]
            w_init = tf.random_uniform([n_in, action_dim], minval=-3e-3, maxval=3e-3)
            output, output_weights = fc_layer(h, action_dim, tf.nn.tanh, w_init=w_init, name='mu', phase=self.training_phase)
            network.add_layer(output, output_weights)
            scaled_output = tf.multiply(output, action_bounds, name='mu_scaled')
            network.add_layer(scaled_output)

            return scaled_output, x, network

    def predict(self, observations, phase=True):
        """
        Predicts the actions using actor network.

        :param observations: Tensor observations
        :param phase: train=True, test=False
        :return: Tensor actions
        """
        return self.sess.run(self.output, {
            self.observations: observations,
            self.training_phase: phase
        })

    def predict_target(self, observations, phase=True):
        """
        Predicts the actions using TARGET actor network.

        :param observations: Tensor observations
        :param phase: train=True, test=False
        :return: Tensor actions
        """
        return self.sess.run(self.target_output, {
            self.target_observations: observations,
            self.training_phase: phase
        })

    def train(self, observations, action_gradients, phase=True):
        """
        Trains the actor network using policy gradient as described in 'DDPG' class.

        :param observations: Tensor observations
        :param phase: train=True, test=False
        :param action_gradients: Tensor action gradients calculated by critic network
        """
        self.sess.run(self.optim, {
            self.observations: observations,
            self.action_gradients: action_gradients,
            self.training_phase: phase
        })

    def init_target_net(self):
        """
        Initializes the target actor network parameters to be equal to the actor network parameters.
        """
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        """
        Performs soft target update according to 'update_target_net_op'
        """
        self.sess.run(self.update_target_net_op)

    def print_summary(self):
        """
        Print the summary of actor network. Is the same as target network so only one has to printed.
        """
        self.network.print_summary()

