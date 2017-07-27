# To run this script you need to specify the mode (train or test). This is passed as an argument.
# Example: python DQN_Mountain_Car.py train | or | python DQN_Mountain_Car.py test

import tensorflow as tf
import gym
import sys

from drl.algorithms import DQN
from drl.nn import Losses
from drl.utilities import color_print

if __name__ == '__main__':
	mode = None
	try:
		mode = sys.argv[1]
	except IndexError:
		color_print("Incorrect usage. You need to specify the mode (train or test) you want to use this script in.",
		            color='red', mode='bold')
		color_print("Correct usage: 'python DQN_Mountain_Car.py train' or 'python DQN_Mountain_Car.py test'",
		            color='blue')

	env = gym.make('MountainCar-v0')
	n_actions = env.action_space.n
	n_obs = env.observation_space.shape[0]
	sess = tf.InteractiveSession()
	dqn = DQN(sess, env, n_actions, n_obs)

	dqn.dqn_options.SCALE_OBSERVATIONS = True
	dqn.dqn_options.SCALING_LIST = [100, 10000]
	dqn.dqn_options.BATCH_SIZE = 128

	dqn.training_network.options.LOSS_TYPE = Losses.HUBER_LOSS
	dqn.target_network.options.LOSS_TYPE = Losses.HUBER_LOSS

	dqn.training_network.create_network()
	dqn.target_network.create_network()
	dqn.initialize()
	sess.run(tf.global_variables_initializer())

	if mode == 'train':
		dqn.populate_replay_buffer(100000, action_set='reduced', allowed_actions=[0, 2], random_repeat_n=10)
		dqn.train(2000, -200, action_set='reduced', allowed_actions=[0, 2], save_path='tmp_MC', save_freq=100, random_repeat_n=10)

	elif mode == 'test':
		dqn.dqn_options.CURRENT_EPSILON = 0
		dqn.test('tmp_MC')
