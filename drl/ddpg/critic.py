import tensorflow as tf

from drl.utilities import fc_layer, bn_layer, TFNetwork


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

        # Boolean saying for phase of system, train=True, test=False
        self.phase = tf.placeholder(dtype=tf.bool, name='phase')

        # Construct model for critic network
        self.output, self.observations, self.actions, self.network = self._build_model('critic', obs_dim, action_dim)

        # Construct model for target critic network
        self.target_output, self.target_observations, self.target_actions, self.target_network = \
            self._build_model('target_critic', obs_dim, action_dim)

        # Set weight variables
        self.weights = self.network.get_weights()
        self.target_weights = self.target_network.get_weights()

        # OP target network weight init
        self.init_target_net_op = [self.target_weights[i].assign(self.weights[i]) for i in range(len(self.target_weights))]

        # OP for soft target update of target critic network
        self.update_target_net_op = [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                                     for i in range(len(self.weights))]

        # OP for calculating action gradient for policy gradient update of actor
        self.action_gradients_op = tf.gradients(self.output, self.actions)

        # OP for updating critic
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_target')
        loss = tf.losses.mean_squared_error(self.y_target, self.output)
        regularizers = [tf.nn.l2_loss(self.weights[i]) for i in range(len(self.weights))]
        self.loss = loss + self.l2_param*tf.reduce_sum(regularizers)
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_model(self, name, obs_dim, action_dim):
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
        with tf.variable_scope(name):
            network = TFNetwork(name)
            num_layers = len(self.hidden_nodes)
            if not (num_layers >= 2):
                raise Exception("Critic network needs at least two hidden layers")

            # Placeholder for observations
            x = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim], name='observations')
            network.add_layer(x)

            h = x
            # Set layer_func to Fully-Connected or Batch-Normalization layer
            layer_func = fc_layer
            if self.batch_norm:
                layer_func = bn_layer

            # First layer with with only observations as input
            h, h_weights = layer_func(h, self.hidden_nodes[0], tf.nn.relu, layer_idx=0, phase=self.phase)
            network.add_layer(h, h_weights)

            # Placeholder for actions
            u = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name='actions')
            network.add_layer(u)

            # Second layer with actions added
            h = tf.concat([h, u], axis=1)
            network.add_layer(h)

            # Hidden layers
            for i in range(1, num_layers):
                h, h_weights = layer_func(h, self.hidden_nodes[i], tf.nn.relu, layer_idx=i, phase=self.phase)
                network.add_layer(h, h_weights)

            # Output layer
            n_in = h.get_shape().as_list()[1]
            w_init = tf.random_uniform([n_in, 1], minval=-3e-3, maxval=3e-3)
            output, h_weights = fc_layer(h, 1, w_init=w_init, name='Q', phase=self.phase)
            network.add_layer(output, h_weights)

            return output, x, u, network

    def predict(self, observations, actions, phase=True):
        """
        Predicts the Q-values using critic network.

        :param observations: Tensor observations
        :param actions: Tensor actions
        :param phase: train=True, test=False
        :return: Tensor Q-values
        """
        return self.sess.run(self.output, {
            self.observations: observations,
            self.actions: actions,
            self.phase: phase
        })

    def predict_target(self, observations, actions, phase=True):
        """
        Predicts the Q-values using TARGET critic network.

        :param observations: Tensor observations
        :param actions: Tensor actions
        :param phase: train=True, test=False
        :return: Tensor Q-values
        """
        return self.sess.run(self.target_output, {
            self.target_observations: observations,
            self.target_actions: actions,
            self.phase: phase
        })

    def train(self, observations, actions, y_target, phase=True):
        """
        Trains the critic network by minimizing loss as described in 'DDPG' class.

        :param observations: Tensor observations
        :param actions: Tensor actions
        :param y_target: Tensor targets (y(i))
        :param phase: train=True, test=False
        :return: loss
        """
        _, loss, q = self.sess.run([self.optim, self.loss, self.output], {
            self.observations: observations,
            self.actions: actions,
            self.y_target: y_target,
            self.phase: phase
        })
        return loss, q

    def action_gradients(self, observations, actions, phase=True):
        """
        Function for calculating 'action_gradients' using 'action_gradients_op'

        :param observations: Tensor observations
        :param actions: Tensor actions
        :param phase: train=True, test=False
        :return: Tensor action_gradients
        """
        return self.sess.run(self.action_gradients_op, {
            self.observations: observations,
            self.actions: actions,
            self.phase: phase
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
        """
        Print the summary of actor network. Is the same as target network so only one has to printed.
        """
        self.network.print_summary()