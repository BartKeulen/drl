import numpy as np
import tensorflow as tf
from collections import OrderedDict
from drl.utilities import *


def uniform_fan_in(h_in, n_out):
    n_in = h_in.get_shape().as_list()[1]
    return tf.random_uniform([n_in, n_out],
                             minval=-1 / np.sqrt(n_in),
                             maxval=1 / np.sqrt(n_in))


def fc_layer(h_in, n_out, activation=None, w_init=None, i=None, name=None, phase=None):
    if w_init is None:
        w_init = uniform_fan_in(h_in, n_out)

    w_name = 'Weight'
    b_name = 'bias'
    if name is None:
        name = 'hidden_layer'
    if i is not None:
        w_name += '_{:s}'.format(str(i))
        b_name += '_{:s}'.format(str(i))
        name += '_{:s}'.format(str(i))

    W = tf.Variable(w_init, name=w_name)
    b = tf.Variable(tf.zeros(shape=[n_out]), name=b_name)
    a = tf.matmul(h_in, W) + b
    if activation is None:
        h = tf.identity(a, name=name)
    else:
        h = activation(a, name=name)
    return h, [W, b]


def bn_layer(h_in, n_out, activation=None, w_init=None, i=None, name=None, phase=None, decay=0.999, epsilon=1e-5):
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

    # TODO: Add support for testing (keeping moving average of mean and variance)
    ema = tf.train.ExponentialMovingAverage(decay=0.999)

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

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.layers = OrderedDict()

    def add_layer(self, layer, weights=None):
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
        weights = []
        for name, layer in self.layers.items():
            if layer['weights'] is not None:
                for weight in layer['weights']:
                    weights.append(weight)
        return weights

    def print_summary(self):
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

