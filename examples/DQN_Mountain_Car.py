import numpy as np
import gym
import tensorflow as tf
import csv

from drl.dqn import DQN

mode = 'train'

if __name__ == '__main__':
	env_name = 'MountainCar-v0'
	env = gym.make(env_name)

	n_actions = env.action_space.n
	n_obs = env.observation_space.shape[0]

	n_episodes = 1000

	sess = tf.InteractiveSession()
	options = {}

	if mode == 'train':
		options = {
			'batch_size': 32,  # No. of training cases over each SGD update
			'replay_memory_size': 100000,  # SGD updates sampled from this number of most recent frames
			'discount_factor': 0.95,  # Gamma used in Q-learning update
			'learning_rate': 0.03,  # Learning rate used by RMSProp
			'initial_epsilon': 1,  # Initial value of epsilon in epsilon-greedy exploration
			'final_epsilon': 0.1,  # Final value of epsilon in epsilon-greedy exploration
			'final_exploration_frame': 50000 # No. of frames over which initial value of epsilon is linearly annealed to it's final value
		}
	elif mode == 'test':
		options = {
			'initial_epsilon': 0
		}

	dqn_network_options = {
		'n_fc': 3,  # Number of fully-connected layers
		'fc_units': [64, 128, 64]  # Number of output units in each fully-connected layer
	}

	n_parameter_updates = 0
	target_network_update_freq = 100

	dqn = DQN(sess, n_actions, n_obs, options, dqn_network_options)

	sess.run(tf.global_variables_initializer())

	if mode == 'train':
		for episode in range(n_episodes):
			obs = env.reset()
			done = False
			episode_reward = 0
			t = 0
			while not done:
				# env.render()
				action = dqn.select_action(obs)
				action_gym = np.argmax(action)
				next_obs, reward, done, info = env.step(action_gym)
				t += 1

				dqn.store_transition(obs, action, reward, done, next_obs)
				episode_reward += reward

				obs = next_obs

				dqn.update_epsilon()

				minibatch = dqn.sample_minibatch()
				dqn.gradient_descent_step(minibatch)
				n_parameter_updates += 1

				if(n_parameter_updates%target_network_update_freq == 0):
					dqn.update_target_network()

				if done:
					if(episode%50 == 0):
						dqn.save('tmp_MC/training.ckpt')
					epsilon = dqn.get_epsilon()
					loss = dqn.get_loss()
					with open('tmp_MC/training_log.csv', 'a', newline='') as csvfile:
						writer = csv.writer(csvfile, delimiter=',')
						writer.writerow([episode, t, episode_reward, epsilon, loss])
					print("Episode: {},  Time Steps: {}, Episode reward: {}, Epsilon: {:.3f}, Loss: {:.3f}".format(episode, t, episode_reward, epsilon, loss))
					break

	elif mode == 'test':
		dqn.restore('tmp_MC/training.ckpt')

		obs = env.reset()
		done = False
		episode_reward = 0
		t = 0
		while not done:
			env.render()
			action = dqn.select_action(obs)
			action_gym = np.argmax(action)
			next_obs, reward, done, info = env.step(action_gym)
			t += 1

			episode_reward += reward
			obs = next_obs
