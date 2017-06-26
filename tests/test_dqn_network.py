from unittest import TestCase
import tensorflow as tf
from random import randint

from drl.dqn import DQNNetwork

class TestDQNNetwork(TestCase):

    def setUp(self):
        self.sess = tf.InteractiveSession()
        self.num_actions, self.options = self.generate_random_options()
        self.network = DQNNetwork(self.num_actions, self.options)
        self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()

    def test_create_network(self):
        n_of_Q_values = self.extract_n_of_Q_Values(self.network.get_Q_Value())
        self.assertEqual(n_of_Q_values, self.num_actions)
        self.assertEqual(self.n_layers, self.network.get_number_of_layers())

    def extract_n_of_Q_Values(self, Q_value):
        Q_value_str = str(Q_value)
        start = Q_value_str.find('shape=(?, ') + 10
        end = Q_value_str.find('), dtype', start)
        n_of_Q_values = Q_value_str[start:end]
        return int(n_of_Q_values)

    def generate_random_options(self):
        num_actions = randint(5, 100)
        n_conv = randint(1, 10)
        n_fc = randint(1, 5)

        # Additional +3 for input, output and reshape or flattened layer when going from convolutional layer to dense layer
        self.n_layers = n_conv + n_fc + 3

        available_filter_sizes = [8, 16, 32, 64, 128, 256, 512]
        available_kernel_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        available_conv_strides = [1, 2, 3, 4]

        available_fc_unit_sizes = [128, 256, 512, 1024, 2048, 4096]

        conv_filters = []
        conv_kernel_sizes = []
        conv_strides = []

        fc_units = []

        for i in range(n_conv):
            filter_size = randint(0, len(available_filter_sizes) - 1)
            kernel_size = randint(0, len(available_kernel_sizes) - 1)
            stride_size = randint(0, len(available_conv_strides) - 1)

            conv_filters.append(available_filter_sizes[filter_size])
            conv_kernel_sizes.append(available_kernel_sizes[kernel_size])
            conv_strides.append(available_conv_strides[stride_size])

        for i in range(n_fc):
            fc_unit_size = randint(0, len(available_fc_unit_sizes) - 1)

            fc_units.append(available_fc_unit_sizes[fc_unit_size])

        options = {
            'n_conv': n_conv,                           # Number of convolutional layers
            'conv_filters': conv_filters,               # Number of filters in each convolutional layer
            'conv_kernel_sizes': conv_kernel_sizes,     # Kernel sizes for each of the convolutional layer
            'conv_strides': conv_strides,               # Stride sizes for each of the convolutional layer

            'n_fc': n_fc,                               # Number of fully-connected layers
            'fc_units': fc_units                        # Number of output units in each fully-connected layer
        }

        return num_actions, options