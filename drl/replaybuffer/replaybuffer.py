from collections import deque
import random
import numpy as np


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
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        Adds new experience to replay buffer.

            1) Experience is added to left of deque
            2) If full experience is removed from the right of the deque

        :param obs_t: state
        :param action: action
        :param reward: reward
        :param done: terminal
        :param obs_tp1: next state
        """
        experience = (obs_t, action, reward, done, obs_tp1)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """
        :return: number of experiences in replay buffer
        """
        return self.count

    def sample(self, batch_size):
        """
        Samples a random mini-batch from the replay buffer.

        :param batch_size: size of mini-batch
        :return: Array states, Array actions, Array rewards, Array terminals, Array next states
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        """
        Clears the whole replay buffer, count is set to zero.
        """
        self.buffer.clear()
        self.count = 0
