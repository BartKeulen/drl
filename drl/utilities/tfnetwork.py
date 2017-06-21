import numpy as np
import tensorflow as tf
from collections import OrderedDict
from drl.utilities import *


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

