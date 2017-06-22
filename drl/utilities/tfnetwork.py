import numpy as np
import tensorflow as tf
from collections import OrderedDict
from drl.utilities import *


def uniform_fan_in(h_in, n_out):
    """
    Creates a tf value of size [n_in, n_out], where n_in is the number of outputs of layer h_in.
    The values are initialized with a uniform distribution in interval [-1/sqrt(n_in), 1/sqrt(n_in)]

    :param h_in: previous layer
    :param n_out: output size
    :return: Tensorflow random value
    """
    n_in = h_in.get_shape().as_list()[1]
    return tf.random_uniform([n_in, n_out],
                             minval=-1 / np.sqrt(n_in),
                             maxval=1 / np.sqrt(n_in))


def fc_layer(h_in, n_out, activation=None, w_init=None, layer_idx=None, name=None, phase=None):
    """
    Creates a fully connected layer with h_in as previous layer.

        a = h_in*W + b
        h = activation(a)

    :param h_in: previous layer
    :param n_out: number of outputs current layer
    :param activation: activation function to be used (standard None)
    :param w_init: weight initialization (standard uniform_fan_in)
    :param layer_idx: layer index (standard None)
    :param name: name of the layer (standard h)
    :param phase: phase, train=True, test=False
    :return: Output h of fully connected layer, [W, b] layer weights and biases
    """
    if w_init is None:
        w_init = uniform_fan_in(h_in, n_out)

    w_name = 'Weight'
    b_name = 'bias'
    if name is None:
        name = 'hidden_layer'
    if layer_idx is not None:
        w_name += '_{:s}'.format(str(layer_idx))
        b_name += '_{:s}'.format(str(layer_idx))
        name += '_{:s}'.format(str(layer_idx))

    W = tf.Variable(w_init, name=w_name)
    b = tf.Variable(tf.zeros(shape=[n_out]), name=b_name)
    a = tf.matmul(h_in, W) + b
    if activation is None:
        h = tf.identity(a, name=name)
    else:
        h = activation(a, name=name)
    return h, [W, b]


def bn_layer(h_in, n_out, activation=None, w_init=None, i=None, name=None, phase=None, decay=0.999, epsilon=1e-5):
    """
    Creates a batch normalization layer with h_in as previous layer.

    Batch normalization is implemented as in 'Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift, Sergey Ioffe and Christian Szegedy'

        a = h_in*W + b
        bn = gamma * (a - mean(a)) / sqrt(var(a) + epsilon) + beta
        h = activation(bn)

    During training the mean and variance are calculated directly form the mini-batch. A population mean and variance
    are being kept by applying an exponential moving average filter over the batch means and variances. The population
    mean and variance are used during testing/inference.

    :param h_in: previous layer
    :param n_out: number of outputs current layer
    :param activation: activation function to be used (standard None)
    :param w_init: weight initialization (standard uniform_fan_in)
    :param layer_idx: layer index (standard None)
    :param name: name of the layer (standard h)
    :param phase: phase, train=True, test=False
    :param decay:
    :return: Output h of fully connected layer, [W, beta, gamma] layer weights and biases
    """
    if phase is None:
        raise Exception("Batch normalization layer requires the phase of the system as input to the layer. \
         The phase is a tf.placeholder(tf.bool) with 'training=True' and 'test=False'")

    if w_init is None:
        w_init = uniform_fan_in(h_in, n_out)

    w_name = 'Weight'
    beta_name = 'beta'
    gamma_name = 'gamma'
    if name is None:
        name = 'hidden_layer'
    if i is not None:
        w_name += '_{:s}'.format(str(i))
        beta_name += '_{:s}'.format(str(i))
        gamma_name += '_{:s}'.format(str(i))
        name += '_{:s}'.format(str(i))

    W = tf.Variable(w_init,
                    name=w_name)

    a = tf.matmul(h_in, W)

    beta = tf.Variable(tf.zeros(shape=[n_out]), name=beta_name)
    gamma = tf.Variable(tf.ones(shape=[n_out]), name=gamma_name)

    pop_mean = tf.Variable(tf.zeros(shape=[1, n_out]), trainable=False)
    pop_var = tf.Variable(tf.ones(shape=[1, n_out]), trainable=False)

    def if_training():
        mean, var = tf.nn.moments(a, [0], keep_dims=True)
        train_mean = tf.assign(pop_mean, pop_mean * decay + mean * (1-decay))
        train_var = tf.assign(pop_var, pop_var * decay + var * (1-decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(a, mean, var, beta, gamma, epsilon)

    def if_test():
        return tf.nn.batch_normalization(a, pop_mean, pop_var, beta, gamma, epsilon)

    bn = tf.cond(phase, if_training, if_test)

    if activation is None:
        with tf.variable_scope(name):
            h = bn
    else:
        h = activation(bn, name=name)
    return h, [W, beta, gamma]


class TFNetwork(object):
    """
    Helper object for storing neural network information, i.e. layers and corresponding weights.
    """

    def __init__(self, name):
        """
        Constructs a TFNetwork object.

        :param name: name for the network, e.g. critic
        """
        self.name = name
        self.layers = OrderedDict()

    def add_layer(self, layer, weights=None):
        """
        Add a layer to the network. For correct printing purposes the layers need to be added in correct order.

        :param layer: layer to be added
        :param weights: weights of the layer (default is None, e.g. inputs don't have any weights)
        """
        self.layers[layer.name] = {
            'tensor': layer
        }
        weight_info = None
        if weights is not None:
            weight_info = weights
        self.layers[layer.name].update({
            'weights': weight_info
        })

    def get_weights(self):
        """
        :return: array containing all the weights of the network.
        """
        weights = []
        for name, layer in self.layers.items():
            if layer['weights'] is not None:
                for weight in layer['weights']:
                    weights.append(weight)
        return weights

    def print_summary(self):
        """
        Prints a clear summary of the complete network. Based on the print_summary results of Keras.
        The summary contains a row for each layer, with the row displaying the layer name, shape, params and weights.

        Example with name=critic:

                                                             CRITIC
            --------------------------------------------------------------------------------------------------------
            Layer                            Shape                Params       Weights
            ========================================================================================================
            critic/observations:0            (None, 3)
            --------------------------------------------------------------------------------------------------------
            critic/hidden_layer_0:0          (None, 400)          2000         critic/Weight_0:0: (3, 400)
                                                                               critic/beta_0:0: (400)
                                                                               critic/gamma_0:0: (400)
            --------------------------------------------------------------------------------------------------------
            critic/actions:0                 (None, 1)
            --------------------------------------------------------------------------------------------------------
            critic/concat:0                  (None, 401)
            --------------------------------------------------------------------------------------------------------
            critic/hidden_layer_1:0          (None, 300)          120900       critic/Weight_1:0: (401, 300)
                                                                               critic/beta_1:0: (300)
                                                                               critic/gamma_1:0: (300)
            --------------------------------------------------------------------------------------------------------
            critic/Q:0                       (None, 1)            301          critic/Weight:0: (300, 1)
                                                                               critic/bias:0: (1)
            --------------------------------------------------------------------------------------------------------
            Total                                                 123201
            --------------------------------------------------------------------------------------------------------
        """

        lengths = [0]*4
        for name, layer in self.layers.items():

            if len(name) > lengths[0]:
                lengths[0] = len(name)
            shape_str = tensor_shape_string(layer['tensor'])

            if len(shape_str) > lengths[1]:
                lengths[1] = len(shape_str)

            if layer['weights'] is not None:
                num_params = 0
                for weight in layer['weights']:
                    shape_str = tensor_shape_string(weight)
                    num_params += tensor_num_params(weight)

                    if len(weight.name + shape_str) > lengths[3]:
                        lengths[3] = len(weight.name + shape_str)

                lengths[2] = len(str(num_params))

        tot_length = 0
        for i in range(4):
            lengths[i] += 10
            tot_length += lengths[i]
        length_to_weights = tot_length - lengths[3]

        len_to_name = int((tot_length + len(self.name)) / 2)
        print('\n\033[1m{:>{w_name}s}\033[0m'.format(self.name.upper(), w_name=len_to_name))
        print('-' * tot_length)
        print('\033[1m{:{w_name}s}{:{w_shape}s}{:{w_params}s}{:{w_weights}s}\033[0m'
              .format('Layer', 'Shape', 'Params', 'Weights',
                      w_name=lengths[0], w_shape=lengths[1], w_params=lengths[2], w_weights=lengths[3]))
        print('=' * tot_length)

        tot_params = 0
        for name, layer in self.layers.items():
            if layer['weights'] is not None:
                num_params = 0
                weight_strs = []
                for weight in layer['weights']:
                    num_params += tensor_num_params(weight)

                    shape_str = tensor_shape_string(weight)
                    weight_strs.append('{:s}: {:s}'.format(weight.name, shape_str))
                tot_params += num_params

                shape_str = tensor_shape_string(layer['tensor'])
                layer_str = '{:{w_name}s}{:{w_shape}s}{:<{w_params}d}{:{w_weights}s}'\
                    .format(name, shape_str, num_params, weight_strs[0],
                            w_name=lengths[0], w_shape=lengths[1], w_params=lengths[2], w_weights=lengths[3])
                for i in range(1, len(weight_strs)):
                    layer_str += '\n{:{w_to_weights}s}{:{w_weights}s}'.format('', weight_strs[i],
                                                                              w_to_weights=length_to_weights, w_weights=lengths[3])
            else:
                shape_str = tensor_shape_string(layer['tensor'])
                layer_str = '{:{w_name}s}{:{w_shape}s}'\
                    .format(name, shape_str, w_name=lengths[0], w_shape=lengths[1])

            print(layer_str)
            print('-' * tot_length)
        print('{:{w_name}s}{:{w_shape}s}{:<{w_params}d}'
              .format('Total', '', tot_params, w_name=lengths[0], w_shape=lengths[1], w_params=lengths[2]))
        print('-' * tot_length)

