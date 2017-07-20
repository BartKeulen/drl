# To run this script you need to specify the mode (train or test). This is passed as an argument.
# Example: python DQN_Mountain_Car.py train | or | python DQN_Mountain_Car.py test

import tensorflow as tf
import gym
import sys

from drl.dqn import DQN, DQN_Options


def display_error_message():
	colors = {
		'red': '\033[91m',
		'blue': '\033[94m',
		'reset': '\033[0m'
	}

	print(
		colors['red'] + "Incorrect usage. You need to specify the mode (train or test) you want to use this script in.")
	print("\nCorrect usage:")
	print(colors['blue'] + "python {} train".format(sys.argv[0]) + colors['red'])
	print("OR")
	print(colors['blue'] + "python {} test".format(sys.argv[0]) + colors['reset'])
	exit()


if __name__ == '__main__':
	mode = None
	try:
		mode = sys.argv[1]
	except IndexError:
		display_error_message()

	env_name = 'MountainCar-v0'
	env = gym.make(env_name)
	n_actions = env.action_space.n
	n_obs = env.observation_space.shape[0]
	allowed_actions = [0, 2]

	DQN_Options.SCALE_OBSERVATIONS = True
	DQN_Options.SCALING_LIST = [100, 10000]

	if mode == 'test':
		DQN_Options.CURRENT_EPSILON = 0

	sess = tf.InteractiveSession()
	dqn = DQN(sess, env, n_actions, n_obs)
	sess.run(tf.global_variables_initializer())

	if mode == 'train':
		dqn.populate_replay_buffer(50000, action_set='reduced', allowed_actions=[0, 2])
		dqn.train(2000, -200, action_set='reduced', allowed_actions=[0, 2], save_path='tmp_MC', save_freq=100)

	elif mode == 'test':
		dqn.test('tmp_MC')
