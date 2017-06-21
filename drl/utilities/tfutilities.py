import numpy as np
import tensorflow as tf


def uniform_fan_in(h_in, n_out):
    n_in = h_in.get_shape().as_list()[1]
    return tf.random_uniform([n_in, n_out],
                             minval=-1 / np.sqrt(n_in),
                             maxval=1 / np.sqrt(n_in))


def fc_layer(h_in, n_out, activation=None, w_init=None, i=None, name=None):
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


def bn_layer(h_in, n_out, activation=None, training_phase=True, w_init=None, i=None, name=None):
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

    mean, var = tf.nn.moments(a, [0], keep_dims=True)
    beta = tf.Variable(tf.zeros(shape=[n_out]), name=beta_name)
    gamma = tf.Variable(tf.ones(shape=[n_out]), name=gamma_name)
    bn = tf.nn.batch_normalization(a, mean, var, beta, gamma, 1e-5)

    if activation is None:
        with tf.variable_scope(name):
            h = a
    else:
        h = activation(bn, name=name)
    return h, [W, beta, gamma]


def tensor_shape_string(tensor):
    shape = tensor.get_shape().as_list()
    shape_str = '('
    for i in range(len(shape)):
        shape_str += '{:s}'.format(str(shape[i]))
        if i < len(shape)-1:
            shape_str += ', '
    shape_str += ')'
    return shape_str


def tensor_num_params(tensor):
    shape = tensor.get_shape().as_list()
    num_params = 1
    for i in range(len(shape)):
        if shape[i] is None:
            raise Exception("Can only calculate number of params when size is fixed, e.g. no tensor with shape [None, 10]")
        num_params *= shape[i]
    return num_params


def print_network_summary(name, layers, weights):
    print('\033[1mSummary {:s} network:\033[0m'.format(name))
    for layer in layers:
        shape_str = tensor_shape_string(layer)
        print('{:25s}     {:10s}'.format(layer.name, shape_str))
    print('')
    for weight in weights:
        shape_str = tensor_shape_string(weight)
        num_params = tensor_num_params(weight)
        print('{:25s}     {:10s}      {:10d}'.format(weight.name, shape_str, num_params))
    print('')
