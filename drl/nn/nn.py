import tensorflow as tf
from drl.utilities import print_dict, color_print, print_table
from drl.utilities import tfutilities


# noinspection PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8
class NN_Options(object):
	N_FC = 2  # Number of fully-connected layers.
	FC_UNITS = [64, 64]  # Number of output units in each fully-connected layer.

	ACTIVATION_TYPE = 'relu'  # Activation function for all your layers.
	LOSS_TYPE = 'mse'  # Loss function you would like to use.
	BATCH_NORM = False  # Switch Batch Normalization on/off. By default it is off.
	DROPOUT = False  # Switch dropout on/off. By default it is off.
	DROPOUT_LAYERS = []  # List containing layer numbers where you want dropout.
	KEEP_PROB = None  # Percentage of neurons to keep on.

	def get_all_options_dict(self):
		dict = {
			'n_fc': self.N_FC,
			'fc_units': self.FC_UNITS,
			'activation_type': self.ACTIVATION_TYPE,
			'loss_type': self.LOSS_TYPE,
			'batch_norm': self.BATCH_NORM,
			'dropout': self.DROPOUT,
			'dropout_layers': self.DROPOUT_LAYERS,
			'keep_prob': self.KEEP_PROB
		}
		return dict


# noinspection PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8
class ConvNet_Options(object):
	"""
	Default options are set to the atari convnet used in:
	    'Human-level control through deep reinforcement learning, Volodymyr Mnih, et al.' - https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
	    'Playing Atari with Deep Reinforcement Learning, Volodymyr Mnih, et al.' - https://arxiv.org/pdf/1312.5602.pdf
	    Lua code: https://sites.google.com/a/deepmind.com/dqn/
	"""
	IMAGE_SIZE = 84  # Square image dimensions.
	CHANNELS = 4  # Number of image channels.

	N_CONV = 3  # Number of convolutional layers.
	CONV_FILTERS = [32, 64, 64]  # Number of filters in each convolutional layer.
	CONV_KERNEL_SIZES = [8, 4, 3]  # Kernel sizes for each of the convolutional layer.
	CONV_STRIDES = [4, 2, 1]  # Stride sizes for each of the convolutional layer.

	N_FC = 1  # Number of fully-connected layers.
	FC_UNITS = [512]  # Number of output units in each fully-connected layer.

	ACTIVATION_TYPE = 'relu'  # Activation function for all your layers.
	LOSS_TYPE = 'mse'  # Loss function you would like to use.
	POOLING = False  # Switch Pooling on/off. By default it is off.
	POOLING_TYPE = 'max_pooling'  # Type of pooling you want to apply. Default is set to max pooling.
	POOLING_LAYERS = []  # List containing layer numbers where you want pooling to be applied.
	POOLING_KSIZE = 2  # Kernel size to be used for the pooling operation.
	POOLING_STRIDE = 2  # Stride size to be used for the pooling operation.
	BATCH_NORM = False  # Switch Batch Normalization on/off. By default it is off.
	DROPOUT = False  # Switch dropout on/off. By default it is off.
	DROPOUT_LAYERS = []  # List containing layer numbers where you want dropout.
	KEEP_PROB = None  # Percentage of neurons to keep on.

	def get_all_options_dict(self):
		dict = {
			'image_size': self.IMAGE_SIZE,
			'channels': self.CHANNELS,

			'n_conv': self.N_CONV,
			'conv_filters': self.CONV_FILTERS,
			'conv_kernel_sizes': self.CONV_KERNEL_SIZES,
			'conv_strides': self.CONV_STRIDES,

			'n_fc': self.N_FC,
			'fc_units': self.FC_UNITS,

			'activation_type': self.ACTIVATION_TYPE,
			'loss_type': self.LOSS_TYPE,
			'pooling': self.POOLING,
			'pooling_type': self.POOLING_TYPE,
			'pooling_layers': self.POOLING_LAYERS,
			'pooling_kernel_size': self.POOLING_KSIZE,
			'pooling_stride': self.POOLING_STRIDE,
			'batch_norm': self.BATCH_NORM,
			'dropout': self.DROPOUT,
			'dropout_layers': self.DROPOUT_LAYERS,
			'keep_prob': self.KEEP_PROB
		}
		return dict


# noinspection PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8
class Losses(object):
	MEAN_SQUARE_ERROR = 'mse'
	ABSOLUTE_DIFFERENCE = 'abs_diff'
	HUBER_LOSS = 'huber'
	HINGE_LOSS = 'hinge'
	LOG_LOSS = 'log'
	SOFTMAX_CROSS_ENTROPY = 'softmax_cross_entropy'
	SIGMOID_CROSS_ENTROPY = 'sigmoid_cross_entropy'
	SPARSE_SOFTMAX_CROSS_ENTROPY = 'sparse_softmax_cross_entropy'

	@classmethod
	def get_all_losses(cls):
		return ['MEAN_SQUARE_ERROR', 'ABSOLUTE_DIFFERENCE', 'HUBER_LOSS', 'HINGE_LOSS', 'LOG_LOSS',
		        'SOFTMAX_CROSS_ENTROPY', 'SIGMOID_CROSS_ENTROPY', 'SPARSE_SOFTMAX_CROSS_ENTROPY']


# noinspection PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8
class Activations(object):
	RECTIFIED_LINEAR = 'relu'
	RECTIFIED_LINEAR6 = 'relu6'
	CONCATENATED_RELU = 'crelu'
	EXPONENTIAL_LINEAR = 'elu'
	SOFTPLUS = 'softplus'
	SOFTSIGN = 'softsign'
	SIGMOID = 'sigmoid'
	TANH = 'tanh'
	LINEAR = 'linear'

	@classmethod
	def get_all_activations(cls):
		return ['RECTIFIED_LINEAR', 'RECTIFIED_LINEAR6', 'CONCATENATED_RELU', 'EXPONENTIAL_LINEAR', 'SOFTPLUS',
		        'SOFTSIGN', 'SIGMOID', 'TANH', 'LINEAR']


# noinspection PyPep8,PyPep8,PyPep8,PyPep8
class Pooling(object):
	MAX_POOLING = 'max_pooling'
	AVERAGE_POOLING = 'avg_pooling'

	@classmethod
	def get_all_pooling(cls):
		return ['MAX_POOLING', 'AVERAGE_POOLING']


# noinspection PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8,PyPep8
class NN(object):
	def __init__(self, n_actions, n_obs=None, network_type=None, network_name=None):
		"""
		Constructs 'NN' object.

			:param n_actions: number of actions
			:param n_obs: number of observations
			:param network_type: 'conv' or 'fc'
			:param network_name: gives the user an option to name networks in order to avoid confusion in
								 case of more than one network
		"""
		self.is_training = tf.placeholder(tf.bool, name='mode')
		self.n_actions = n_actions
		self.n_obs = n_obs
		self.network_type = network_type
		self.network_name = network_name

		if self.network_type == 'conv':
			self.options = ConvNet_Options()
		else:
			self.options = NN_Options()
			self.dropout_layers = self.options.DROPOUT_LAYERS.copy()

			if self.options.DROPOUT:
				if self.options.KEEP_PROB == None:
					color_print("ERROR: You need to set the keep_prob value if using Dropout", color='red', mode='bold')
					exit()
				if len(self.dropout_layers) == 0:
					color_print("NOTE: Using dropout for all dense layers!", color='blue')
					for layer_number in range(self.options.N_FC):
						self.dropout_layers.append(layer_number)
				else:
					self.dropout_layers = [layer_number - 1 for layer_number in self.dropout_layers]

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
		self.weights.append(self.weight_variable(
			[self.kernel_sizes[conv_layer_number], self.kernel_sizes[conv_layer_number],
			 self.filters[conv_layer_number], self.filters[conv_layer_number + 1]],
			'Conv_Weights_' + str(conv_layer_number + 1)))
		self.biases.append(
			self.bias_variable([self.filters[conv_layer_number + 1]], 'Conv_Biases_' + str(conv_layer_number + 1)))

		# Add convolutional layer and apply relu activation to it's output
		conv_layer = tf.nn.conv2d(self.layers[-1], self.weights[-1],
		                          strides=[1, self.strides[conv_layer_number], self.strides[conv_layer_number], 1],
		                          padding='SAME', name='Conv_' + str(conv_layer_number + 1))
		if self.options.BATCH_NORM:
			conv_layer = self.batch_norm(conv_layer, name='Conv_BN_' + str(conv_layer_number + 1))
		conv_layer = self.activation_function(tf.nn.bias_add(conv_layer, self.biases[-1]),
		                                      name='Conv_Activated_' + str(conv_layer_number + 1))

		if self.options.POOLING:
			conv_layer = self.pooling(conv_layer, name='Conv_Pooling_' + str(conv_layer_number + 1))

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
		self.weights.append(
			self.weight_variable([self.fc_units[dense_layer_number], self.fc_units[dense_layer_number + 1]],
			                     'FC_Weights_' + str(dense_layer_number + 1)))
		self.biases.append(
			self.bias_variable([self.fc_units[dense_layer_number + 1]], 'FC_Biases_' + str(dense_layer_number + 1)))

		# Add fully connected layer and apply relu activation to it's output
		dense = tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1],
		               name='FC_' + str(dense_layer_number + 1))
		if self.options.BATCH_NORM:
			dense = self.batch_norm(dense, name='FC_BN_' + str(dense_layer_number + 1))
		dense = self.activation_function(dense, name='FC_Activated_' + str(dense_layer_number + 1))
		if self.options.DROPOUT and dense_layer_number in self.dropout_layers:
			dense = self.dropout(dense, name='FC_Dropout_' + str(dense_layer_number + 1))

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

	def loss_function(self, target, predicted):
		"""
		Returns the loss function as selected by the user. The default loss function used is mean_square.
		
			:param target Target Value
			:param predicted Predicted Value

			:return: Loss function
		"""
		if self.options.LOSS_TYPE == 'mse':
			return tf.losses.mean_squared_error(target, predicted)
		elif self.options.LOSS_TYPE == 'abs_diff':
			return tf.losses.absolute_difference(target, predicted)
		elif self.options.LOSS_TYPE == 'huber':
			return tf.losses.huber_loss(target, predicted)
		elif self.options.LOSS_TYPE == 'hinge':
			return tf.losses.hinge_loss(target, predicted)
		elif self.options.LOSS_TYPE == 'log':
			return tf.losses.log_loss(target, predicted)
		elif self.options.LOSS_TYPE == 'softmax_cross_entropy':
			return tf.losses.softmax_cross_entropy(target, predicted)
		elif self.options.LOSS_TYPE == 'sigmoid_cross_entropy':
			return tf.losses.sigmoid_cross_entropy(target, predicted)
		elif self.options.LOSS_TYPE == 'sparse_softmax_cross_entropy':
			return tf.losses.sparse_softmax_cross_entropy(target, predicted)
		else:
			color_print("ERROR: Please select a loss function that is supported by this library!", color='red',
			            mode='bold')
			color_print("We don't support: {}".format(self.options.LOSS_TYPE), color='red')
			print()
			color_print(
				"This library supports the following loss functions. You can access them by using 'Losses.YOUR_DESIRED_LOSS_FUNCTION'",
				color='blue')
			print_table([Losses.get_all_losses()], ['Supported Losses'], indentation='center', color='blue')
			exit()

	def activation_function(self, input_layer, name=None):
		"""
		Applies the activation function as selected by the user. The default activation function used is relu.

			:param input_layer: Layer that is fed as input to the activation function.
			:param name: Optional Parameter. Allows you to name the layer.
			:return: Activation applied to the input layer.
		"""
		if self.options.ACTIVATION_TYPE == 'relu':
			return tf.nn.relu(input_layer, name=name)
		elif self.options.ACTIVATION_TYPE == 'relu6':
			return tf.nn.relu6(input_layer, name=name)
		elif self.options.ACTIVATION_TYPE == 'crelu':
			return tf.nn.crelu(input_layer, name=name)
		elif self.options.ACTIVATION_TYPE == 'elu':
			return tf.nn.elu(input_layer, name=name)
		elif self.options.ACTIVATION_TYPE == 'softplus':
			return tf.nn.softplus(input_layer, name=name)
		elif self.options.ACTIVATION_TYPE == 'softsign':
			return tf.nn.softsign(input_layer, name=name)
		elif self.options.ACTIVATION_TYPE == 'sigmoid':
			return tf.sigmoid(input_layer, name=name)
		elif self.options.ACTIVATION_TYPE == 'tanh':
			return tf.tanh(input_layer, name=name)
		elif self.options.ACTIVATION_TYPE == 'linear':
			return input_layer
		else:
			color_print("ERROR: Please select an activation function that is supported by this library!", color='red',
			            mode='bold')
			print()
			color_print(
				"This library supports the following activation functions. You can access them by using 'Activations.YOUR_DESIRED_ACTIVATION_FUNCTION'",
				color='blue')
			print_table([Activations.get_all_activations()], ['Supported Activations'], indentation='center',
			            color='blue')
			exit()

	def pooling(self, input_layer, name=None):
		"""
		Applies the type of pooling as selected by the user. The default pooling type used is max_pooling.

			:param input_layer: Layer that is fed as input for applying pooling
			:param name: Optional Parameter. Allows you to name the layer.
			:return: Pooling applied to the input layer.
		"""
		if self.options.POOLING_TYPE == 'max_pooling':
			return tf.nn.max_pool(input_layer,
			                      ksize=[1, self.options.POOLING_KSIZE, self.options.POOLING_KSIZE, 1],
			                      strides=[1, self.options.POOLING_STRIDE, self.options.POOLING_STRIDE, 1],
			                      padding='SAME', name=name)
		elif self.options.POOLING_TYPE == 'avg_pooling':
			return tf.nn.avg_pool(input_layer,
			                      ksize=[1, self.options.POOLING_KSIZE, self.options.POOLING_KSIZE, 1],
			                      strides=[1, self.options.POOLING_STRIDE, self.options.POOLING_STRIDE, 1],
			                      padding='SAME', name=name)
		else:
			color_print("ERROR: Please select a pooling operation that is supported by this library!", color='red',
			            mode='bold')
			print()
			color_print(
				"This library supports the following pooling functions. You can access them by using 'Pooling.YOUR_DESIRED_POOLING_TYPE",
				color='blue')
			print_table([Pooling.get_all_pooling()], ['Supported Pooling Types'], indentation='center', color='blue')
			exit()

	def batch_norm(self, input_layer, decay=0.999, epsilon=1e-3, name=None):
		"""
		Batch Normalization wrapper.
		Original implementation: https://r2rt.com/implementing-batch-normalization-in-tensorflow.html

			:param input_layer: Input layer to which we will apply BN
			:param decay: Decay for the moving average. Reasonable values for decay are close to 1.0.
			:param epsilon: Small float added to variance to avoid dividing by zero.
			:param name: Optional Parameter. Allows you to name the layer.
			:return: Batch normalized layer output.
		"""
		scale = tf.Variable(tf.ones([input_layer.get_shape()[-1]]))
		beta = tf.Variable(tf.zeros([input_layer.get_shape()[-1]]))
		pop_mean = tf.Variable(tf.zeros([input_layer.get_shape()[-1]]), trainable=False)
		pop_var = tf.Variable(tf.ones([input_layer.get_shape()[-1]]), trainable=False)

		def train():
			batch_mean, batch_var = tf.nn.moments(input_layer, [0])
			train_mean = tf.assign(pop_mean,
			                       pop_mean * decay + batch_mean * (1 - decay))
			train_var = tf.assign(pop_var,
			                      pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(input_layer,
				                                 batch_mean, batch_var, beta, scale, epsilon, name=name)

		def test():
			return tf.nn.batch_normalization(input_layer,
			                                 pop_mean, pop_var, beta, scale, epsilon, name=name)

		return tf.cond(self.is_training, train, test)

	def dropout(self, input_layer, name=None):
		"""
		Returns dropout with user-defined keep_prob if training. Uses keep_prob of 1.0 if testing.

			:param input_layer Input layer to which we will apply dropout
			:param name Optional Parameter. Allows you to name the layer.
			:return: dropout layer
		"""

		def train():
			return tf.nn.dropout(input_layer, self.options.KEEP_PROB, name=name)

		def test():
			return tf.nn.dropout(input_layer, 1.0, name=name)

		return tf.cond(self.is_training, train, test)

	def create_network(self):
		"""
		Creates a custom neural net based on user-defined options.
		"""
		if self.network_type == 'conv':
			self.image_size = self.options.IMAGE_SIZE.copy()
			self.channels = self.options.CHANNELS.copy()
			# Placeholder for Input image/s (Data format: [batch_size, height, width, channels]
			self.input = tf.placeholder("float", [None, self.image_size, self.image_size, self.channels],
			                            name='Input_Layer')

			# Get required settings from options
			self.kernel_sizes = self.options.CONV_KERNEL_SIZES.copy()
			self.filters = self.options.CONV_FILTERS.copy()
			self.strides = self.options.CONV_STRIDES.copy()
			self.fc_units = self.options.FC_UNITS.copy()

			# Add channels for 1st layer
			self.filters.insert(0, self.channels)
		else:
			# Placeholder for Input
			self.input = tf.placeholder("float", [None, self.n_obs], name='Input_Layer')

			self.fc_units = self.options.FC_UNITS.copy()
			self.fc_units.insert(0, self.n_obs)

		self.weights = []
		self.biases = []
		self.layers = [self.input]

		if self.network_type == 'conv':
			# Add convolutional layers
			for conv_layer_number in range(self.options.N_CONV):
				self.layers.append(self.conv2d(conv_layer_number))

			# Reshape or flatten the last convolutional layer
			self.layers.append(self.reshape())

			# Add dense or fully connected layers
			for dense_layer_number in range(self.options.N_FC):
				self.layers.append(self.dense(dense_layer_number))
		else:
			# Add dense or fully connected layers
			for dense_layer_number in range(self.options.N_FC):
				self.layers.append(self.dense(dense_layer_number))

		# Finally add the output layer
		self.layers.append(self.output_layer())

		# Set Q value to output of neural network
		self.set_output_value(self.layers[-1])

		# Set total number of layers in the layer stack
		self.set_number_of_layers(len(self.layers))

		self.target_value = tf.placeholder(shape=[None], dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32)

		# Expected Q value
		self.expected_value = tf.reduce_sum(tf.multiply(self.output_value, self.actions), axis=1)

		# Compute loss
		self.loss = self.loss_function(self.target_value, self.expected_value)

		# Print network summary
		self.print_network_summary()

	def get_output_value(self):
		return self.output_value

	def set_output_value(self, Q_value):
		self.output_value = Q_value

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
		print_dict(self.network_name + " Network options: ", self.options.get_all_options_dict())

	def print_network_summary(self):
		pass
		# tfutilities.print_network_summary(self.network_name, self.layers[1:], self.weights)
