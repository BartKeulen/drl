import tensorflow as tf
import numpy as np
import site

from Atari import Atari

from drl.dqn import DQN

save_path = '/home/shadylady/Akshat/Reinforcement Learning/Programs/drl/examples/tmp/'

if __name__ == '__main__':
	path = str.encode(str(site.getsitepackages()[0]) + "/atari_py/atari_roms/") # Convert string to bytes format
	atari = Atari(path + b"breakout.bin")
	actions = atari.legal_actions

	sess = tf.InteractiveSession()

	dqn = DQN(sess, actions)

	sess.run(tf.global_variables_initializer())

	episodes = 10
	T = 10000
	update_time = 100

	for episode in range(episodes):
		state = atari.newGame()
		# 4 most recent frames experienced by the agent are given as input to the Q Network
		state = np.stack((state, state, state, state), axis=2).reshape((84, 84, 4))

		episode_reward = 0

		for t in range(T):
			action = dqn.select_action(state)

			next_state, reward, game_over = atari.next(action)
			next_state = np.append(next_state, state, axis=2)[:, :, 1:]

			episode_reward += reward

			dqn.store_transition(state, action, reward, game_over, next_state)

			minibatch = dqn.sample_minibatch()

			dqn.gradient_descent_step(minibatch)

			state = next_state

			print("\rEpisode: {}, Time Step: {}, Episode Reward: {}, Current Reward: {}".format(episode, t, episode_reward, reward), end="")

			dqn.update_epsilon()

			if t%update_time == 0:
				dqn.update_target_network()
		print()
		dqn.save(save_path, episode)
