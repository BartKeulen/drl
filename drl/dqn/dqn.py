import tensorflow as tf
import numpy as np
import random

from drl.replaybuffer import ReplayBuffer
from drl.dqn import DQNNetwork
from drl.utilities import print_dict

# Algorithm info
info = {
    'name': 'DQN'
}

# Algorithm options
options = {
    'batch_size': 32,                           # No. of training cases over each SGD update
    'replay_memory_size': 1000000,              # SGD updates sampled from this number of most recent frames
    'discount_factor': 0.99,                    # Gamma used in Q-learning update
    'learning_rate': 0.00025,                   # Learning rate used by RMSProp
    'gradient_momentum': 0.95,                  # Gradient momentum used by RMSProp
    'min_squared_gradient': 0.01,               # Constant added to the squared gradient in denominator of RMSProp update
    'initial_epsilon': 1,                       # Initial value of epsilon in epsilon-greedy exploration
    'final_epsilon': 0.1,                       # Final value of epsilon in epsilon-greedy exploration
    'final_exploration_frame': 1000000,         # No. of frames over which initial value of epsilon is linearly annealed to it's final value
}

class DQN(object):
    """
    Implementation of the deep Q-learning with experience replay algorithm from

        'Human-level control through deep reinforcement learning, Volodymyr Mnih, et al.' - https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
        'Playing Atari with Deep Reinforcement Learning, Volodymyr Mnih, et al.' - https://arxiv.org/pdf/1312.5602.pdf
    """
    def __init__(self,
                 sess,
                 env,
                 actions,
                 options_in=None):
        """
        Constructs 'DQN' object.

        :param sess: Tensorflow session
        :param env: environment
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
        self._env = env
        self.actions = actions
        self.n_actions = len(actions)

        # Update options
        if options_in is not None:
            options.update(options_in)

        self.batch_size = options['batch_size'].copy()
        self.discount_factor = options['discount_factor']
        self.learning_rate = options['learning_rate']
        self.gradient_momentum = options['gradient_momentum']
        self.min_squared_gradient = options['min_squared_gradient']
        self.initial_epsilon = options['initial_epsilon']
        self.final_epsilon = options['final_epsilon']
        self.final_exploration_frame = options['final_exploration_frame']

        print_dict("Algorithm options:", options)

        self.replay_buffer = ReplayBuffer(options['replay_memory_size'])
        self.training_network = DQNNetwork(self.n_actions)
        self.target_network = DQNNetwork(self.n_actions)
        self.epsilon = self.initial_epsilon

        self._sess.run(tf.global_variables_initializer())

    def select_action(self, current_state):
        action = np.zeros(self.n_actions)

        if random.random() < self.epsilon:
            index = random.randrange(0, self.n_actions)
        else:
            index = np.argmax(self.training_network.get_Q_Value().eval(feed_dict = {self.training_network.input: [current_state]}))

        action[index] = 1.0

        return action

    def store_transition(self, state, action, reward, terminal, new_state):
        self.replay_buffer.add(state, action, reward, terminal, new_state)

    def sample_minibatch(self):
        return self.replay_buffer.sample_batch(self.batch_size)

    def gradient_descent_step(self, minibatch):
        for state, action, reward, terminal, new_state in minibatch:
            target = reward

            if not terminal:
                target = reward + self.discount_factor * np.amax(self.target_network.get_Q_Value().eval(feed_dict = {self.target_network.input: [new_state]}))

            self.train = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.gradient_momentum, epsilon=self.min_squared_gradient).minimize(self.training_network.loss)
            self._sess.run(self.train, feed_dict = {self.training_network.input: [state], self.training_network.target_Q_Value: target, self.training_network.actions: self.actions})

    def update_target_network(self):
        weights = self.training_network.get_weights()
        biases = self.training_network.get_weights()

        self.target_network.set_weights(weights)
        self.target_network.set_biases(biases)

    def save(self, path, global_step=None):
        saver = tf.train.Saver()
        save_path = saver.save(self._sess, save_path=path, global_step=global_step)
        print('Network Parameters saved in file:\n {:s}'.format(save_path))

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self._sess, path)
        print('Network Parameters restored from file:\n {:s}'.format(path))