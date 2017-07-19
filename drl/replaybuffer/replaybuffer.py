import numpy as np
from sklearn.neighbors import KernelDensity


class ReplayBuffer(object):
    """
    Data structure for an experience replay buffer.

    Experiences are added one at a time to the buffer. The buffer uses FIFO to replace experiences when the maximum
    capacity is reached
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
        np.random.seed(random_seed)

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


class ReplayBufferKD(ReplayBuffer):

    def __init__(self, size):
        super(ReplayBufferKD, self).__init__(size)

    def kd_estimate(self, tree_size, samples):
        if tree_size > super(ReplayBufferKD, self).size():
            tree_size = super(ReplayBufferKD, self).size()
        obses_t = self.sample(tree_size)[0]

        kd = KernelDensity()
        kd.fit(obses_t)

        scores = kd.score_samples(samples)

        return scores

