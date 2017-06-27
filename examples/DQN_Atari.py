import tensorflow as tf
import numpy as np
import site

from Atari import Atari

from drl.dqn import DQN

if __name__ == '__main__':
	path = str.encode(str(site.getsitepackages()[0]) + "/atari_py/atari_roms/") # Convert string to bytes format
	atari = Atari(path + b"breakout.bin")
	actions = atari.legal_actions

	sess = tf.InteractiveSession()

	dqn = DQN(sess, actions)

	sess.run(tf.global_variables_initializer())

	episodes = 1
	T = 10000
	update_time = 100

	for episode in range(episodes):
		state = atari.newGame()
		# 4 most recent frames experienced by the agent are given as input to the Q Network
		state = np.stack((state, state, state, state), axis=2).reshape((84, 84, 4))

		for t in range(1, T):
			action = dqn.select_action(state)

			next_state, reward, game_over = atari.next(action)
			next_state = np.append(next_state, state, axis=2)[:, :, 1:]

			dqn.store_transition(state, action, reward, game_over, next_state)

			minibatch = dqn.sample_minibatch()

			dqn.gradient_descent_step(minibatch)

			state = next_state

			print("Time Step: {}, Reward: {}".format(t, reward))

			dqn.update_epsilon()

			if t%update_time == 0:
				sess.run(dqn.update_target_network())
