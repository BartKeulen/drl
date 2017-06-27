import tensorflow as tf
from drl.utilities import print_dict
from drl.utilities import tfutilities

IMAGE_SIZE = 84
CHANNELS = 4

"""
Default options are set to the atari convnet used in:
    'Human-level control through deep reinforcement learning, Volodymyr Mnih, et al.' - https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
    'Playing Atari with Deep Reinforcement Learning, Volodymyr Mnih, et al.' - https://arxiv.org/pdf/1312.5602.pdf
    Lua code: https://sites.google.com/a/deepmind.com/dqn/
"""
options = {
    'n_conv': 3,                        # Number of convolutional layers
    'conv_filters': [32, 64, 64],       # Number of filters in each convolutional layer
    'conv_kernel_sizes': [8, 4, 3],     # Kernel sizes for each of the convolutional layer
    'conv_strides': [4, 2, 1],          # Stride sizes for each of the convolutional layer

    'n_fc': 1,                          # Number of fully-connected layers
    'fc_units':[512]                    # Number of output units in each fully-connected layer
}

class DQNNetwork(object):

    def __init__(self, n_actions, network_name='DQNNetwork', options_in=None):
        """
        Constructs 'DQNNetwork' object.

            :param n_actions: number of actions
            :param network_name: gives the user an option to name networks in order to avoid confusion in
                                 case of more than one network
            :param options_in: available and default options for DQNNetwork object:

                'n_conv': 3,                        # Number of convolutional layers
                'conv_filters': [32, 64, 64],       # Number of filters in each convolutional layer
                'conv_kernel_sizes': [8, 4, 3],     # Kernel sizes for each of the convolutional layer
                'conv_strides': [4, 2, 1],          # Stride sizes for each of the convolutional layer

                'n_fc': 1,                          # Number of fully-connected layers
                'fc_units':[512]                    # Number of output units in each fully-connected layer
        """
        self.n_actions = n_actions
        self.network_name = network_name

        if options_in is not None:
            options.update(options_in)

        self.print_options()

        self.create_network()

    def weight_variable(self, shape, name, stddev=0.1, mean=0, seed=None):
        """
        Generates random initial weights for a layer.

            :param shape: shape of weights
            :param name: if you want to add a name to recognize which layer this set of weights belong to
            :param stddev: standard deviation of normal distribution
            :param mean: mean of normal distribution
            :param seed: seed for randomness
            :return: weights
        """
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, mean=mean, seed=seed), name=name)

    def bias_variable(self, shape, name, value=0.1):
        """
        Generates initial bias for a layer.

            :param shape: shape of biases
            :param name: if you want to add a name to recognize which layer this set of biases belong to
            :param value: value the biases are set to
            :return: biases
        """
        return tf.Variable(tf.constant(value=value, shape=shape), name=name)

    def conv2d(self, conv_layer_number):
        """
        Creates a 2D convolution layer.

            :param conv_layer_number: layer number of the convolutional layer
            :return: a convolutional layer
        """
        # Add weights and biases for each convolutional layer
        self.weights.append(self.weight_variable([self.kernel_sizes[conv_layer_number], self.kernel_sizes[conv_layer_number], self.filters[conv_layer_number], self.filters[conv_layer_number + 1]], 'Conv_Weights_' + str(conv_layer_number + 1)))
        self.biases.append(self.bias_variable([self.filters[conv_layer_number + 1]], 'Conv_Biases_' + str(conv_layer_number + 1)))

        # Add convolutional layer and apply relu activation to it's output
        conv_layer = tf.nn.conv2d(self.layers[-1], self.weights[-1], strides=[1, self.strides[conv_layer_number], self.strides[conv_layer_number], 1], padding='SAME')
        conv_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, self.biases[-1]), name='Conv_' + str(conv_layer_number+1))

        return conv_layer

    def reshape(self):
        """
        Reshape or flatten out layer. Used when going from convolutional layers to dense layers.

            :return: flattened output of convolutional layer
        """

        # Reshape for fully connected or dense layer
        conv_shape = self.layers[-1].get_shape().as_list()
        reshape = tf.reshape(self.layers[-1], [-1, conv_shape[1] * conv_shape[2] * conv_shape[3]], name='Flatten_Layer')

        self.fc_units.insert(0, conv_shape[1] * conv_shape[2] * conv_shape[3])

        return reshape

    def dense(self, dense_layer_number):
        """
        Creates a dense or fully connected layer.

            :param dense_layer_number: layer number of the dense or fully connected layer
            :return: a dense or fully connected layer
        """
        # Add weights and biases for each fully connected layer
        self.weights.append(self.weight_variable([self.fc_units[dense_layer_number], self.fc_units[dense_layer_number + 1]], 'FC_Weights_' + str(dense_layer_number + 1)))
        self.biases.append(self.bias_variable([self.fc_units[dense_layer_number + 1]], 'FC_Biases_' + str(dense_layer_number + 1)))

        # Add fully connected layer and apply relu activation to it's output
        dense = tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1])
        dense = tf.nn.relu(dense, name='FC_' + str(dense_layer_number + 1))

        return dense

    def output_layer(self):
        """
        Generates the final or output layer of the network.

            :return: output layer
        """
        # Weights and biases for Last Hidden Layer
        self.weights.append(self.weight_variable([self.fc_units[-1], self.n_actions], name='Last_Hidden_Layer_Weights'))
        self.biases.append(self.bias_variable([self.n_actions], name='Last_Hidden_Layer_Bias'))

        output = tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1], name='Output_Layer')

        return output

    def create_network(self):
        """
        Creates a convnet based on user-defined options.
        If no options are given by user, it by default generates the convnet used by Deepmind for their Atari application.
        """

        # Placeholder for Input image/s
        self.input = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], name='Input_Layer')

        # Get required settings from options
        self.kernel_sizes = options['conv_kernel_sizes'].copy()
        self.filters = options['conv_filters'].copy()
        self.strides = options['conv_strides'].copy()
        self.fc_units = options['fc_units'].copy()

        self.weights = []
        self.biases = []
        self.layers = [self.input]

        # Add channels for 1st layer
        self.filters.insert(0, CHANNELS)

        # Add convolutional layers
        for conv_layer_number in range(options['n_conv']):
            self.layers.append(self.conv2d(conv_layer_number))

        # Reshape or flatten the last convolutional layer
        self.layers.append(self.reshape())

        # Add dense or fully connected layers
        for dense_layer_number in range(options['n_fc']):
            self.layers.append(self.dense(dense_layer_number))

        # Finally add the output layer
        self.layers.append(self.output_layer())

        # Set Q value to output of neural network
        self.set_Q_Value(self.layers[-1])

        # Set total number of layers in the layer stack
        self.set_number_of_layers(len(self.layers))

        # Get prediction (highest action value)
        self.predict = tf.argmax(self.get_Q_Value(), 1)

        self.target_Q_Value = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.n_actions, dtype=tf.float32)

        # Expected Q value
        self.Q = tf.reduce_sum(tf.multiply(self.Q_value, self.actions_onehot), axis=1)

        # Compute loss
        self.loss = tf.square(self.target_Q_Value - self.Q)

        # Print network summary
        self.print_network_summary()

    def get_Q_Value(self):
        return self.Q_value

    def set_Q_Value(self, Q_value):
        self.Q_value = Q_value

    def set_number_of_layers(self, n_layers):
        self.n_layers = n_layers

    def get_number_of_layers(self):
        return self.n_layers

    def set_layers(self, layers):
        self.layers = layers

    def get_layers(self):
        return self.layers

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_biases(self, biases):
        self.biases = biases

    def get_biases(self):
        return self.biases

    def print_options(self):
        print_dict(self.network_name + " Network options: ", options)

    def print_network_summary(self):
        tfutilities.print_network_summary(self.network_name, self.layers, self.weights)