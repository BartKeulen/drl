import numpy as np
import tensorflow as tf





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
