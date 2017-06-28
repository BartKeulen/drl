import tensorflow as tf
import numpy as np
import site
import csv

from Atari import Atari

from drl.dqn import DQN

save_path = 'tmp/training.ckpt'

def populate_replay_buffer(dqn,replay_start_size):
	current_replay_size = 0

	while(current_replay_size < replay_start_size):
		state = atari.newGame()
		# 4 most recent frames experienced by the agent are given as input to the Q Network
		state = np.stack((state, state, state, state), axis=2).reshape((84, 84, 4))

		for t in range(T):
			if (current_replay_size % 100 == 0):
				print("\rPopulating replay buffer. Current Size: {}, Target Size: {}".format(current_replay_size, replay_start_size), end="")

			action = dqn.select_action(state)

			next_state, reward, game_over = atari.next(action)
			next_state = np.append(next_state, state, axis=2)[:, :, 1:]

			dqn.store_transition(state, action, reward, game_over, next_state)
			current_replay_size+=1

			if game_over:
				break

	print()

if __name__ == '__main__':
	path = str.encode(str(site.getsitepackages()[0]) + "/atari_py/atari_roms/") # Convert string to bytes format
	atari = Atari(path + b"breakout.bin")
	actions = atari.legal_actions

	sess = tf.InteractiveSession()

	options = {
		'learning_rate': 0.00025
	}

	dqn = DQN(sess, actions, options)

	sess.run(tf.global_variables_initializer())

	dqn.restore(save_path)

	episodes = 100000
	T = 10000
	update_time = 10000
	number_of_parameter_updates = 0
	update_frequency = 4
	final_exploration_frame = 1000000
	number_of_frames = 0
	replay_start_size = 50000
	action_repeat = 4
	game_over = False

	populate_replay_buffer(dqn, replay_start_size)

	for episode in range(episodes):
		state = atari.newGame()
		# 4 most recent frames experienced by the agent are given as input to the Q Network
		state = np.stack((state, state, state, state), axis=2).reshape((84, 84, 4))

		episode_reward = 0

		for t in range(T):
			action = dqn.select_action(state)

			for i in range(action_repeat):
				next_state, reward, game_over = atari.next(action)
				next_state = np.append(next_state, state, axis=2)[:, :, 1:]
				number_of_frames += 1

				dqn.store_transition(state, action, reward, game_over, next_state)
				episode_reward += reward

				state = next_state

			if(t%update_frequency == 0):
				minibatch = dqn.sample_minibatch()

				dqn.gradient_descent_step(minibatch)
				number_of_parameter_updates += 1

			print("\rEpisode: {}, Time Step: {}, Episode Reward: {}, Current Reward: {}, Loss: {}".format(episode, t, episode_reward, reward, dqn.get_loss()), end="")

			if number_of_parameter_updates%update_time == 0:
				dqn.update_target_network()

			if(number_of_frames%final_exploration_frame == 0):
				dqn.update_epsilon()

			if game_over:
				if (episode % 50 == 0):
					dqn.save(save_path)
					with open('tmp/training_log.csv', 'a', newline='') as csvfile:
						writer = csv.writer(csvfile, delimiter=',')
						writer.writerow([episode, t, episode_reward, reward, dqn.get_loss()])
				break
		print()