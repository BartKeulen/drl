import tensorflow as tf
import numpy as np
import random
import os
from gym.monitoring import VideoRecorder
import datetime
import csv

from drl.replaybuffer import ReplayBuffer
from drl.nn import NN
from drl.utilities import print_dict, color_print

# Algorithm info
info = {
    'name': 'DQN'
}


class DQN_Options(object):
    NETWORK_TYPE = None  # Replace by 'conv' for convolution
    BATCH_SIZE = 32  # No. of training cases over each SGD update
    REPLAY_MEMORY_SIZE = 1000000  # SGD updates sampled from this number of most recent frames
    DISCOUNT_FACTOR = 0.99  # Gamma used in Q-learning update
    LEARNING_RATE = 0.00025  # Learning rate used by optimizer
    INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy exploration
    CURRENT_EPSILON = INITIAL_EPSILON  # Initialize current epsilon to initial value
    FINAL_EPSILON = 0.01  # Final value of epsilon in epsilon-greedy exploration
    FINAL_EXPLORATION_FRAME = 100000  # No. of frames over which initial value of epsilon is linearly annealed to it's final value
    TARGET_NETWORK_UPDATE_FREQUENCY = 10000  # No. of parameter updates after which you should update the target network
    SCALE_OBSERVATIONS = False  # Set to true if you wish to scale your observations
    SCALING_LIST = []  # List containing scale that you want to apply to each observation

    def get_all_options_dict(self):
        dict = {
            'network_type': self.NETWORK_TYPE,
            'batch_size': self.BATCH_SIZE,
            'replay_memory_size': self.REPLAY_MEMORY_SIZE,
            'discount_factor': self.DISCOUNT_FACTOR,
            'learning_rate': self.LEARNING_RATE,
            'initial_epsilon': self.INITIAL_EPSILON,
            'current_epsilon': self.CURRENT_EPSILON,
            'final_epsilon': self.FINAL_EPSILON,
            'final_exploration_frame': self.FINAL_EXPLORATION_FRAME,
            'target_network_update_freq': self.TARGET_NETWORK_UPDATE_FREQUENCY,
            'scale_observations': self.SCALE_OBSERVATIONS,
            'scaling_list': self.SCALING_LIST
        }
        return dict


class DQN(object):
    """
    Implementation of the deep Q-learning with experience replay algorithm from

        'Human-level control through deep reinforcement learning, Volodymyr Mnih, et al.' - https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
        'Playing Atari with Deep Reinforcement Learning, Volodymyr Mnih, et al.' - https://arxiv.org/pdf/1312.5602.pdf
    """

    def __init__(self, sess, env, n_actions, n_obs=None):
        """
        Constructs 'DQN' object.

        :param sess: Tensorflow session
        :param env: current environment
        :param n_actions: number of actions
        :param n_obs: number of observations
        """
        self.sess = sess
        self.env = env
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.dqn_options = DQN_Options()
        print_dict("DQN Algorithm options:", self.dqn_options.get_all_options_dict())

        self.replay_buffer = ReplayBuffer(self.dqn_options.REPLAY_MEMORY_SIZE)
        self.training_network = NN(self.n_actions, n_obs=self.n_obs, network_type=self.dqn_options.NETWORK_TYPE,
                                   network_name='Training')
        self.target_network = NN(self.n_actions, n_obs=self.n_obs, network_type=self.dqn_options.NETWORK_TYPE,
                                 network_name='Target')

        self.n_parameter_updates = 0
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.dqn_options.LEARNING_RATE).minimize(
            self.training_network.loss)
        print('Using Adam Optimizer!')

        self.timestamp = str(datetime.datetime.utcnow())

    def select_action(self, current_state):
        """
        Selects action to perform.
        Selects a random action with probability epsilon, otherwise selects action with max action value.

            :param current_state: current state the agent is in.
            :return: action
        """
        action = np.zeros(self.n_actions)

        if random.random() < self.dqn_options.CURRENT_EPSILON:
            index = random.randrange(0, self.n_actions)
        else:
            index = np.argmax(self.training_network.get_output_value().eval(
                feed_dict={self.training_network.input: [current_state], self.training_network.is_training: False}))

        action[index] = 1.0

        return action

    def select_reduced_action(self, current_state, allowed_actions):
        """
        Selects action to perform from a reduced set of actions.

            :param current_state: current state the agent is in.
            :param allowed_actions: list of actions that are allowed. Example: Say we have 3 actions (0, 1, 2). Then if allowed_actions = [0, 2], will never allow the agent to perform action 1.
            :return: action from the list of allowed action.
        """
        action = self.select_action(current_state)
        while np.argmax(action) not in allowed_actions:
            action = self.select_action(current_state)
        return action

    def store_transition(self, state, action, reward, new_state, terminal):
        """
        Stores transition in replay buffer.

            :param state: state the agent was in
            :param action: action taken by agent
            :param reward: reward received for action performed
            :param terminal: if the agent has reached a terminal state or not
            :param new_state: new state the agent is now in after performing the action
        """
        self.replay_buffer.add(state, action, reward, new_state, terminal)

    def sample_minibatch(self):
        """
        Samples a minibatch of experiences from the replay buffer

            :return: minibatch of experiences
        """
        return self.replay_buffer.sample(self.dqn_options.BATCH_SIZE)

    def parameter_update(self):
        """
        Goes through the sampled minibatch of experiences and sets a target value accordingly.
        Performs SGD.
        """
        states, actions, rewards, new_states, terminal_states = self.sample_minibatch()
        targets = []

        for n in range(len(states)):
            target = rewards[n]
            if not terminal_states[n]:
                target = rewards[n] + self.dqn_options.DISCOUNT_FACTOR * np.amax(
                    self.target_network.get_output_value().eval(
                        feed_dict={self.target_network.input: [new_states[n]], self.target_network.is_training: False}))
            targets.append(target)

        feed_dict = {self.training_network.input: states, self.training_network.target_value: targets,
                     self.training_network.actions: actions, self.training_network.is_training: True}
        train_value, loss_value, q_value = self.sess.run(
            [self.train_step, self.training_network.loss, self.training_network.expected_value], feed_dict=feed_dict)
        self.set_loss(loss_value)

        self.n_parameter_updates += 1
        if (self.n_parameter_updates % self.dqn_options.TARGET_NETWORK_UPDATE_FREQUENCY == 0):
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
        if self.dqn_options.CURRENT_EPSILON > self.dqn_options.FINAL_EPSILON:
            self.dqn_options.CURRENT_EPSILON -= (
                                                    self.dqn_options.INITIAL_EPSILON - self.dqn_options.FINAL_EPSILON) / self.dqn_options.FINAL_EXPLORATION_FRAME

    def scale_observations(self, obs):
        for index in range(len(obs)):
            obs[index] *= self.dqn_options.SCALING_LIST[index]
        return obs

    def populate_replay_buffer(self, size, action_set='all', allowed_actions=None, random_repeat_n=4,
                               print_rate=1000):
        color_print("Populating Replay Buffer. Please wait. Training will start shortly!", color='blue')

        for sample in range(size):
            obs = self.env.reset()
            if self.dqn_options.SCALE_OBSERVATIONS:
                obs = self.scale_observations(obs)
            done = False
            while not done:
                if action_set == 'all':  # If we want to use all actions
                    action = self.select_action(obs)
                elif action_set == 'reduced':  # If we want to use a reduced set of actions
                    action = self.select_reduced_action(obs, allowed_actions)
                action_gym = np.argmax(action)

                # Repeat the action randomly between 1 to random_repeat_n times
                for n_repeat in range(np.random.randint(0, random_repeat_n)):
                    next_obs, reward, done, info = self.env.step(action_gym)
                    if self.dqn_options.SCALE_OBSERVATIONS:
                        next_obs = self.scale_observations(next_obs)
                    self.store_transition(obs, action, reward, next_obs, done)
                    obs = next_obs
                    if done:
                        break
                if sample % print_rate == 0:
                    print("\rReplay Buffer Size: {}".format(sample), end="")
        color_print("\nPopulated Replay Buffer!", color='blue')

    def train(self, episodes, episode_success_threshold, action_set='all', allowed_actions=None, random_repeat_n=4,
              save_path=None, save_freq=None):
        if save_path == None:
            save_path = self.timestamp

        if save_freq == None:
            save_freq = (int)(episodes / 5)

        if not os.path.exists(save_path + '/Videos'):
            os.makedirs(save_path + '/Videos')

        n_consequent_successful_episodes = 0

        for n_episode in range(episodes):
            if (n_episode % save_freq == 0) or (n_episode == episodes - 1):
                video_recorder = VideoRecorder(self.env, save_path + '/Videos/' + str(n_episode) + '.mp4', enabled=True)
                color_print('Recording Video!', color='blue')

            obs = self.env.reset()
            if self.dqn_options.SCALE_OBSERVATIONS:
                obs = self.scale_observations(obs)
            done = False
            episode_reward = 0
            t = 0
            while not done:
                if (n_episode % save_freq == 0) or (n_episode == episodes - 1):
                    video_recorder.capture_frame()
                if action_set == 'all':  # If we want to use all actions
                    action = self.select_action(obs)
                elif action_set == 'reduced':  # If we want to use a reduced set of actions
                    action = self.select_reduced_action(obs, allowed_actions)
                action_gym = np.argmax(action)

                # Repeat the action randomly between 1 to random_repeat_n times
                for n_repeat in range(np.random.randint(0, random_repeat_n)):
                    next_obs, reward, done, info = self.env.step(action_gym)
                    t += 1
                    if self.dqn_options.SCALE_OBSERVATIONS:
                        next_obs = self.scale_observations(next_obs)
                    self.store_transition(obs, action, reward, next_obs, done)
                    episode_reward += reward
                    obs = next_obs

                    if (self.dqn_options.CURRENT_EPSILON == self.dqn_options.FINAL_EPSILON) or done:
                        break
                    else:
                        self.update_epsilon()

                self.parameter_update()

                if done:
                    if (n_episode % save_freq == 0) or (n_episode == episodes - 1):
                        self.save(save_path + '/training.ckpt')
                    epsilon = self.get_epsilon()
                    loss = self.get_loss()
                    with open(save_path + '/training_log.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        writer.writerow([episodes, t, episode_reward, epsilon, loss])
                    print(
                        "Episode: {},  Time Steps: {}, Episode reward: {}, Epsilon: {:.3f}, Loss: {:.3f}, Obs: {}, NCSE: {}".format(
                            n_episode, t, episode_reward, epsilon, loss, obs, n_consequent_successful_episodes))

                    if (episode_reward > episode_success_threshold):
                        n_consequent_successful_episodes += 1
                    else:
                        n_consequent_successful_episodes = 0

            if (n_episode % save_freq == 0) or (n_episode == episodes - 1):
                video_recorder.close()
                color_print("Recording Over!", color='blue')

    def test(self, restore_path=None):
        if restore_path == None:
            restore_path = self.timestamp

        self.restore(restore_path + '/training.ckpt')

        obs = self.env.reset()
        if self.dqn_options.SCALE_OBSERVATIONS:
            obs = self.scale_observations(obs)
        done = False
        episode_reward = 0
        t = 0
        video_recorder = VideoRecorder(self.env, restore_path + '/test_output.mp4', enabled=True)
        while not done:
            video_recorder.capture_frame()
            action = self.select_action(obs)
            action_gym = np.argmax(action)
            next_obs, reward, done, info = self.env.step(action_gym)
            if self.dqn_options.SCALE_OBSERVATIONS:
                next_obs = self.scale_observations(next_obs)
            t += 1
            episode_reward += reward
            obs = next_obs
        color_print('Time taken: {}'.format(t), color='blue')
        video_recorder.close()

    def get_epsilon(self):
        return self.dqn_options.CURRENT_EPSILON

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
        save_path = saver.save(self.sess, save_path=path, global_step=global_step)
        print('Network Parameters saved in file:\n {:s}'.format(save_path))

    def restore(self, path):
        """
        Restores network parameters from a particular checkpoint.
            :param path: file from where the network parameters are restored
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print('Network Parameters restored from file:\n {:s}'.format(path))
