import tensorflow as tf
from drl.utilities import print_dict
from drl.utilities import tfutilities

IMAGE_SIZE = 84
CHANNELS = 1

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

def weight_variable(shape, name, stddev=0.1, mean=0, seed=None):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, mean=mean, seed=seed), name=name)


def bias_variable(shape, name, value=0.1):
    return tf.Variable(tf.constant(value=value, shape=shape), name=name)

class DQNNetwork(object):

    def __init__(self, num_actions, options_in=None):
        """
        Constructs 'DQNNetwork' object.

            :param options_in: available and default options for DQNNetwork object:

                'n_conv': 3,                        # Number of convolutional layers
                'conv_filters': [32, 64, 64],       # Number of filters in each convolutional layer
                'conv_kernel_sizes': [8, 4, 3],     # Kernel sizes for each of the convolutional layer
                'conv_strides': [4, 2, 1],          # Stride sizes for each of the convolutional layer

                'n_fc': 1,                          # Number of fully-connected layers
                'fc_units':[512]                    # Number of output units in each fully-connected layer
        """
        self.num_actions = num_actions

        if options_in is not None:
            options.update(options_in)

        self.print_options()

        self.compute_action_value(num_actions)

    def compute_action_value(self, num_actions):

        # Placeholder for Input image/s
        input = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], name='Input_Layer')

        # Get required settings from options
        kernel_sizes = options['conv_kernel_sizes'].copy()
        filters = options['conv_filters'].copy()
        strides = options['conv_strides'].copy()
        fc_units = options['fc_units'].copy()

        weights = []
        biases = []
        layers = [input]

        # Add channels for 1st layer
        filters.insert(0, CHANNELS)

        for n_conv_layers in range(options['n_conv']):
            # Add weights and biases for each convolutional layer
            weights.append(weight_variable([kernel_sizes[n_conv_layers], kernel_sizes[n_conv_layers], filters[n_conv_layers], filters[n_conv_layers+1]], 'Conv_Weights_' + str(n_conv_layers+1)))
            biases.append(bias_variable([filters[n_conv_layers+1]], 'Conv_Biases_' + str(n_conv_layers+1)))

            # Add convolutional layer and apply relu activation to it's output
            layers.append(tf.nn.conv2d(layers[-1], weights[-1], strides=[1, strides[n_conv_layers], strides[n_conv_layers], 1], padding='SAME', name='Conv_' + str(n_conv_layers+1)))
            layers.append(tf.nn.relu(tf.nn.bias_add(layers[-1], biases[-1]), name='Relu_' + str(n_conv_layers+1)))

        # Reshape for fully connected layer
        conv_shape = layers[-1].get_shape().as_list()
        layers.append(tf.reshape(layers[-1], [-1, conv_shape[1] * conv_shape[2] * conv_shape[3]], name='Flatten_Layer'))

        fc_units.insert(0, conv_shape[1] * conv_shape[2] * conv_shape[3])

        for n_fc_layers in range(options['n_fc']):
            # Add weights and biases for each fully connected layer
            weights.append(weight_variable([fc_units[n_fc_layers], fc_units[n_fc_layers+1]], 'FC_Weights_' + str(n_fc_layers+1)))
            biases.append(bias_variable([fc_units[n_fc_layers+1]], 'FC_Biases_' + str(n_fc_layers+1)))

            # Add fully connected layer and apply relu activation to it's output
            layers.append(tf.add(tf.matmul(layers[-1], weights[-1]), biases[-1], name='FC_' + str(n_fc_layers+1)))
            layers.append(tf.nn.relu(layers[-1], name='FC_Relu_' + str(n_fc_layers+1)))

        # Weights and biases for Last Hidden Layer
        weights.append(weight_variable([fc_units[-1], num_actions], name='Last_Hidden_Layer_Weights'))
        biases.append(bias_variable([num_actions], name='Last_Hidden_Layer_Bias'))

        # Add last hidden layer
        layers.append(tf.add(tf.matmul(layers[-1], weights[-1]), biases[-1], name='Output_Layer'))

        self.set_Q_Value(layers[-1])
        self.set_number_of_layers(int((len(layers) - 3)/2))
        self.set_weights(weights)
        self.set_biases(biases)

        tfutilities.print_network_summary('DQNNetwork', layers, weights)

    def get_Q_Value(self):
        return self.Q_value

    def set_Q_Value(self, Q_value):
        self.Q_value = Q_value

    def set_number_of_layers(self, n_layers):
        self.n_layers = n_layers

    def get_number_of_layers(self):
        return self.n_layers

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_biases(self, biases):
        self.biases = biases

    def get_biases(self):
        return self.biases

    def print_options(self):
        print_dict("Network options: ", options)