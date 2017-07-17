# To run this script you need to specify the mode (train or test). This is passed as an argument.
# Example: python DQN_Mountain_Car.py train | or | python DQN_Mountain_Car.py test

import tensorflow as tf
import gym
from gym.monitoring import VideoRecorder
import numpy as np
import csv
import sys
import os

from drl.dqn import DQN


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

def reduced_action(obs):
	action = [0.0, 1.0, 0.0]
	while(action[1] == 1):
		action = dqn.select_action(obs)
	return action

def populate_replay_buffer(size):
	print('Populating Replay Buffer. Please wait. Training will start shortly!')

	for i in range(size):
		obs = env.reset()
		obs[0] *= 100.0
		obs[1] *= 10000.0
		done = False
		while not done:
			action = reduced_action(obs)
			action_gym = np.argmax(action)
			next_obs, reward, done, info = env.step(action_gym)
			next_obs[0] *= 100.0
			next_obs[1] *= 10000.0

			dqn.store_transition(obs, action, reward, done, next_obs)
			obs = next_obs

			if done:
				if i%1000 == 0:
					print('\rReplay Buffer Size: {}'.format(i), end="")
				break

	print('\nPopulated Replay Buffer!')

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
	options = {}

	if mode == 'test':
		options = {
			'initial_epsilon': 0
		}

	sess = tf.InteractiveSession()
	dqn = DQN(sess, n_actions, n_obs, options)
	sess.run(tf.global_variables_initializer())

	stopping_condition = False
	max_episodes = 5000
	episodes = 0
	n_consequent_successful_episodes = 0

	if mode == 'train':
		if not os.path.exists('tmp_MC_reduced_actions_1/Videos'):
			os.makedirs('tmp_MC_reduced_actions_1/Videos')

		populate_replay_buffer(50000)
		while not stopping_condition:
			if episodes % 100 == 0:
				video_recorder = VideoRecorder(env, 'tmp_MC_reduced_actions_1/Videos/MC_' + str(episodes) + '.mp4', enabled=True)
				print('Recording Video!')

			obs = env.reset()
			obs[0] *= 100.0
			obs[1] *= 10000.0
			done = False
			episode_reward = 0
			t = 0
			while not done:
				if episodes % 100 == 0:
					video_recorder.capture_frame()
				action = reduced_action(obs)
				action_gym = np.argmax(action)
				next_obs, reward, done, info = env.step(action_gym)
				next_obs[0] *= 100.0
				next_obs[1] *= 10000.0
				t += 1

				dqn.store_transition(obs, action, reward, done, next_obs)
				episode_reward += reward

				obs = next_obs

				dqn.parameter_update()
				dqn.update_epsilon()

				if done:
					if (episodes % 50 == 0):
						dqn.save('tmp_MC_reduced_actions_1/training.ckpt')
					epsilon = dqn.get_epsilon()
					loss = dqn.get_loss()
					with open('tmp_MC_reduced_actions_1/training_log.csv', 'a', newline='') as csvfile:
						writer = csv.writer(csvfile, delimiter=',')
						writer.writerow([episodes, t, episode_reward, epsilon, loss])
					print(
						"Episode: {},  Time Steps: {}, Episode reward: {}, Epsilon: {:.3f}, Loss: {:.3f}, Obs: {}, NCSE: {}".format(
							episodes, t, episode_reward, epsilon, loss, obs, n_consequent_successful_episodes))

					if (episode_reward > -200):
						n_consequent_successful_episodes += 1
					else:
						n_consequent_successful_episodes = 0

					if (n_consequent_successful_episodes > 100):
						stopping_condition = True

					break

			if (episodes % 100 == 0):
				video_recorder.close()
				print('Recording Over!')

			episodes += 1
			if (episodes == max_episodes):
				stopping_condition = True
				print("Exceeded allowed limit on number of episodes!")

	elif mode == 'test':
		dqn.restore('tmp_MC_reduced_actions_1/training.ckpt')

		obs = env.reset()
		obs[0] *= 100.0
		obs[1] *= 10000.0
		done = False
		episode_reward = 0
		t = 0
		video_recorder = VideoRecorder(env, 'MC.mp4', enabled=True)
		while not done:
			video_recorder.capture_frame()
			action = dqn.select_action(obs)
			action_gym = np.argmax(action)
			next_obs, reward, done, info = env.step(action_gym)
			next_obs[0] *= 100.0
			next_obs[1] *= 10000.0
			t += 1

			episode_reward += reward
			obs = next_obs
		print('Time taken: {}'.format(t))
		video_recorder.close()