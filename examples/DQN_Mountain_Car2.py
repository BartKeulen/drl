import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import csv
import sys

from drl.dqn import DQN

mode = sys.argv[1]

if __name__ == '__main__':
	env_name = 'MountainCar-v0'
	env = gym.make(env_name)

	n_actions = env.action_space.n
	n_obs = env.observation_space.shape[0]

	sess = tf.InteractiveSession()
	options = {}

	if mode == 'train':
		options = {
			'batch_size': 128,  # No. of training cases over each SGD update
			'replay_memory_size': 1000000,  # SGD updates sampled from this number of most recent frames
			'discount_factor': 0.99,  # Gamma used in Q-learning update
			'learning_rate': 0.00025,  # Learning rate used by RMSProp
			'initial_epsilon': 1,  # Initial value of epsilon in epsilon-greedy exploration
			'final_epsilon': 0.05,  # Final value of epsilon in epsilon-greedy exploration
			'final_exploration_frame': 100000 # No. of frames over which initial value of epsilon is linearly annealed to it's final value
		}
	elif mode == 'test':
		options = {
			'initial_epsilon': 0
		}

	dqn_network_options = {
		'n_fc': 2,  # Number of fully-connected layers
		'fc_units': [32, 32]  # Number of output units in each fully-connected layer
	}

	n_parameter_updates = 0
	target_network_update_freq = 10000

	dqn = DQN(sess, n_actions, n_obs, options, dqn_network_options)

	sess.run(tf.global_variables_initializer())

	stopping_condition = False
	episodes = 0
	n_consequent_successful_episodes = 0

	if mode == 'train':
		monitor = wrappers.Monitor(env, 'tmp_MC3/monitor', video_callable=lambda episode_id: episode_id%100==0, force=True, mode='training')
		while not stopping_condition:
			episodes += 1
			obs = env.reset()
			obs[0] *= 100.0
			obs[1] *= 10000.0
			done = False
			episode_reward = 0
			t = 0
			while not done:
				# env.render()
				action = dqn.select_action(obs)
				action_gym = np.argmax(action)
				next_obs, reward, done, info = env.step(action_gym)
				next_obs[0] *= 100.0
				next_obs[1] *= 10000.0
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
					print('Updated target network!')

				if done:
					if(episodes%50 == 0):
						dqn.save('tmp_MC3/training.ckpt')
					epsilon = dqn.get_epsilon()
					loss = dqn.get_loss()
					with open('tmp_MC3/training_log.csv', 'a', newline='') as csvfile:
						writer = csv.writer(csvfile, delimiter=',')
						writer.writerow([episodes, t, episode_reward, epsilon, loss])
					print("1: Episode: {},  Time Steps: {}, Episode reward: {}, Epsilon: {:.3f}, Loss: {:.3f}, Obs: {}, NCSE: {}".format(episodes, t, episode_reward, epsilon, loss, obs, n_consequent_successful_episodes))

					if(episode_reward > -110):
						n_consequent_successful_episodes += 1
					else:
						n_consequent_successful_episodes = 0

					if(n_consequent_successful_episodes > 100):
						stopping_condition = True

					break

	elif mode == 'test':
		dqn.restore('tmp_MC3/training.ckpt')

		obs = env.reset()
		obs[0] *= 100.0
		obs[1] *= 10000.0
		done = False
		episode_reward = 0
		t = 0
		while not done:
			env.render()
			action = dqn.select_action(obs)
			action_gym = np.argmax(action)
			next_obs, reward, done, info = env.step(action_gym)
			next_obs[0] *= 100.0
			next_obs[1] *= 10000.0
			t += 1

			episode_reward += reward
			obs = next_obs
		print('Time taken: {}'.format(t))