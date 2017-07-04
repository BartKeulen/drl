import tensorflow as tf
import numpy as np
import random

from drl.replaybuffer import ReplayBuffer
from drl.dqn import DQNNetwork
from drl.utilities import print_dict

# Algorithm info
info = {
	'Name': 'DQN'
}

# Algorithm options
dqn_options = {
	'network_type': None,
	'batch_size': 32,                           # No. of training cases over each SGD update
	'replay_memory_size': 1000000,              # SGD updates sampled from this number of most recent frames
	'discount_factor': 0.99,                    # Gamma used in Q-learning update
	'learning_rate': 0.00025,                   # Learning rate used by RMSProp
	'gradient_momentum': 0.95,                  # Gradient momentum used by RMSProp
	'min_squared_gradient': 0.01,               # Constant added to the squared gradient in denominator of RMSProp update
	'initial_epsilon': 1,                       # Initial value of epsilon in epsilon-greedy exploration
	'final_epsilon': 0.01,                      # Final value of epsilon in epsilon-greedy exploration
	'final_exploration_frame': 100000,          # No. of frames over which initial value of epsilon is linearly annealed to it's final value
	'target_network_update_freq': 10000         # No. of parameter updates after which you should update the target network
}

class DQN(object):
	"""
	Implementation of the deep Q-learning with experience replay algorithm from

		'Human-level control through deep reinforcement learning, Volodymyr Mnih, et al.' - https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
		'Playing Atari with Deep Reinforcement Learning, Volodymyr Mnih, et al.' - https://arxiv.org/pdf/1312.5602.pdf
	"""
	def __init__(self,
				 sess,
				 n_actions,
				 n_obs = None,
				 dqn_options_in = None,
	             dqn_network_options = None):
		"""
		Constructs 'DQN' object.

		:param sess: Tensorflow session
		:param actions: available actions
		:param observations: available observations/states
		:param options_in: available and default options for DQN object:

			'batch_size': 32,                           # No. of training cases over each SGD update
			'replay_memory_size': 1000000,              # SGD updates sampled from this number of most recent frames
			'discount_factor': 0.99,                    # Gamma used in Q-learning update
			'learning_rate': 0.00025,                   # Learning rate used by RMSProp
			'gradient_momentum': 0.95,                  # Gradient momentum used by RMSProp
			'min_squared_gradient': 0.01,               # Constant added to the squared gradient in denominator of RMSProp update
			'initial_epsilon': 1,                       # Initial value of epsilon in epsilon-greedy exploration
			'final_epsilon': 0.1,                       # Final value of epsilon in epsilon-greedy exploration
			'final_exploration_frame': 1000000,         # No. of frames over which initial value of epsilon is linearly annealed to it's final value
		"""
		self._sess = sess
		self.n_actions = n_actions
		self.n_obs = n_obs

		# Update options
		if dqn_options_in is not None:
			dqn_options.update(dqn_options_in)

		self.network_type = dqn_options['network_type']
		self.batch_size = dqn_options['batch_size']
		self.discount_factor = dqn_options['discount_factor']
		self.learning_rate = dqn_options['learning_rate']
		self.gradient_momentum = dqn_options['gradient_momentum']
		self.min_squared_gradient = dqn_options['min_squared_gradient']
		self.initial_epsilon = dqn_options['initial_epsilon']
		self.final_epsilon = dqn_options['final_epsilon']
		self.final_exploration_frame = dqn_options['final_exploration_frame']
		self.target_network_update_freq = dqn_options['target_network_update_freq']

		print_dict("DQN Algorithm options:", dqn_options)

		self.replay_buffer = ReplayBuffer(dqn_options['replay_memory_size'])
		self.training_network = DQNNetwork(self.n_actions, n_obs=self.n_obs, network_type=self.network_type, network_name='Training', options_in=dqn_network_options)
		self.target_network = DQNNetwork(self.n_actions, n_obs=self.n_obs, network_type=self.network_type, network_name='Target', options_in=dqn_network_options)

		self.n_parameter_updates = 0
		self.epsilon = self.initial_epsilon

		# self.train = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.gradient_momentum, epsilon=self.min_squared_gradient).minimize(self.training_network.loss)
		# self.train = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.training_network.loss)
		self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.training_network.loss)
		print('Using Adam Optimizer!')


	def select_action(self, current_state):
		"""
		Selects action to perform.
		Selects a random action with probability epsilon, otherwise selects action with max action value

			:param current_state: current state the agent is in
			:return: action
		"""
		action = np.zeros(self.n_actions)

		if random.random() < self.epsilon:
			index = random.randrange(0, self.n_actions)
		else:
			index = np.argmax(self.training_network.get_Q_Value().eval(feed_dict = {self.training_network.input: [current_state]}))

		action[index] = 1.0

		return action

	def store_transition(self, state, action, reward, terminal, new_state):
		"""
		Stores transition in replay buffer.

			:param state: state the agent was in
			:param action: action taken by agent
			:param reward: reward received for action performed
			:param terminal: if the agent has reached a terminal state or not
			:param new_state: new state the agent is now in after performing the action
		"""
		self.replay_buffer.add(state, action, reward, terminal, new_state)

	def sample_minibatch(self):
		"""
		Samples a minibatch of experiences from the replay buffer

			:return: minibatch of experiences
		"""
		return self.replay_buffer.sample_batch(self.batch_size)

	def parameter_update(self):
		"""
		Goes through the sampled minibatch of experiences and sets a target value accordingly.
		Performs SGD using RMSProp.
		"""
		states, actions, rewards, terminal_states, new_states = self.sample_minibatch()
		targets = []

		for n in range(len(states)):
			target = rewards[n]
			if not terminal_states[n]:
				target = rewards[n] + self.discount_factor * np.amax(self.target_network.get_Q_Value().eval(feed_dict = {self.target_network.input: [new_states[n]]}))
			targets.append(target)

		feed_dict = {self.training_network.input: states, self.training_network.target_Q_Value: targets, self.training_network.actions: actions}
		train_value, loss_value, q_value = self._sess.run([self.train, self.training_network.loss, self.training_network.Q], feed_dict = feed_dict)
		self.set_loss(loss_value)

		self.n_parameter_updates += 1
		if(self.n_parameter_updates%self.target_network_update_freq == 0):
			self.update_target_network()
			print('Updated target network!')

	def update_target_network(self):
		"""
		Update target network with training network parameters
		"""
		weights = self.training_network.get_weights()
		biases = self.training_network.get_weights()

		self.target_network.set_weights(weights)
		self.target_network.set_biases(biases)

	def update_epsilon(self):
		"""
		Anneals epsilon from initial value to final value.
		"""
		if self.epsilon > self.final_epsilon:
			self.epsilon -= (self.initial_epsilon - self.final_epsilon)/self.final_exploration_frame

	def get_epsilon(self):
		return self.epsilon

	def set_loss(self, loss_value):
		self.loss_value = loss_value

	def get_loss(self):
		return self.loss_value

	def save(self, path, global_step=None):
		"""
		Creates a checkpoint to save all network parameters.
			:param path: path where network parameters are saved
			:param global_step: If provided the global step number is appended to path to create the checkpoint filename
		"""
		saver = tf.train.Saver()
		save_path = saver.save(self._sess, save_path=path, global_step=global_step)
		print('Network Parameters saved in file:\n {:s}'.format(save_path))

	def restore(self, path):
		"""
		Restores network parameters from a particular checkpoint.
			:param path: file from where the network parameters are restored
		"""
		saver = tf.train.Saver()
		saver.restore(self._sess, path)
		print('Network Parameters restored from file:\n {:s}'.format(path))