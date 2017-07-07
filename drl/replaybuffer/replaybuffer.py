from collections import deque
import random
import numpy as np
from tqdm import tqdm


class ReplayBuffer(object):
    """
    Data structure for implementing experience replay buffer.

    Author: Patrick Emami
    Original: https://github.com/pemami4911/deep-rl/blob/master/ddpg/replay_buffer.py
    """

    def __init__(self, buffer_size, random_seed=123):
        """
        Constructs 'ReplayBuffer' object.
        The right side of the deque contains the most recent experiences.

        :param buffer_size: size of the replay buffer
        :param random_seed: seed for sampling mini-batches
        """
        self._buffer_size = buffer_size
        self._buffer = []
        self._next_idx = 0
        random.seed(random_seed)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        Adds new experience to replay buffer.

            1) Experience is added to left of deque
            2) If full experience is removed from the right of the deque

        :param obs_t: observation
        :param action: action
        :param reward: reward
        :param obs_tp1: next observation
        :param done: terminal
        """
        experience = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx < self._buffer_size:
            self._buffer.append(experience)
        else:
            self._buffer[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._buffer_size

    def size(self):
        """
        :return: number of experiences in replay buffer
        """
        return len(self._buffer)

    def sample(self, batch_size):
        """
        Samples a random mini-batch from the replay buffer.

        :param batch_size: size of mini-batch
        :return: Array observations, Array actions, Array rewards, Array next observations, Array done
        """
        batch_obs_t, batch_action, batch_reward, batch_obs_tp1, batch_done, idxs = [], [], [], [], [], []
        for _ in range(batch_size):
            idx = np.random.randint(self.size())
            experience = self._buffer[idx]
            obs_t, action, reward, obs_tp1, done = experience

            batch_obs_t.append(np.array(obs_t, copy=False))
            batch_action.append(np.array(action, copy=False))
            batch_reward.append(reward)
            batch_obs_tp1.append(np.array(obs_tp1, copy=False))
            batch_done.append(done)
            idxs.append(idx)

        return np.array(batch_obs_t), np.array(batch_action), np.array(batch_reward), np.array(batch_obs_tp1), \
               np.array(batch_done), idxs

    def clear(self):
        """
        Clears the whole replay buffer, count is set to zero.
        """
        self._buffer.clear()
        self._next_idx = 0

