import numpy as np
from sklearn.neighbors import KernelDensity
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer as BLPrioritizedReplayBuffer
import matplotlib.pyplot as plt
from drl.smartexploration.trajectory import Trajectory


class ReplayBuffer(object):
    """
    Data structure for an experience replay buffer.

    Experiences are added one at a time to the buffer. The buffer uses FIFO to replace experiences when the maximum
    capacity is reached
    """

    def __init__(self, buffer_size, random_seed=None):
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
        batch_obs_t, batch_action, batch_reward, batch_obs_tp1, batch_done = [], [], [], [], []
        for _ in range(batch_size):
            idx = np.random.randint(self.size())
            experience = self._buffer[idx]
            obs_t, action, reward, obs_tp1, done = experience

            batch_obs_t.append(np.array(obs_t, copy=False))
            batch_action.append(np.array(action, copy=False))
            batch_reward.append(reward)
            batch_obs_tp1.append(np.array(obs_tp1, copy=False))
            batch_done.append(done)

        return np.array(batch_obs_t), np.array(batch_action), np.array(batch_reward), np.array(batch_obs_tp1), \
               np.array(batch_done)

    def clear(self):
        """
        Clears the whole replay buffer, count is set to zero.
        """
        self._buffer.clear()
        self._next_idx = 0


class ReplayBufferKD(ReplayBuffer):

    def __init__(self,
                 size,
                 kernel='gaussian',
                 bandwidth=0.15,
                 leaf_size=100,):
        super(ReplayBufferKD, self).__init__(size)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.leaf_size = leaf_size

        self.parent_idxes = []
        self.parent = -1

        self.cur_policy = None
        self.policies = []

    def new_episode(self, policy=None):
        self.cur_policy = policy
        self.parent = -1

    def add(self, obs_t, action, reward, obs_tp1, done):
        if self._next_idx < self._buffer_size:
            self.parent_idxes.append(self.parent)
            self.policies.append(self.cur_policy)
        else:
            self.parent_idxes[self._next_idx] = self.parent
            self.policies[self._next_idx] = self.cur_policy
        self.parent = self._next_idx
        super(ReplayBufferKD, self).add(obs_t, action, reward, obs_tp1, done)

    def sample(self, batch_size):
        batch_obs_t, batch_action, batch_reward, batch_obs_tp1, batch_done, idxes, policies = [], [], [], [], [], [], []
        for _ in range(batch_size):
            idx = np.random.randint(self.size())
            experience = self._buffer[idx]
            obs_t, action, reward, obs_tp1, done = experience

            batch_obs_t.append(np.array(obs_t, copy=False))
            batch_action.append(np.array(action, copy=False))
            batch_reward.append(reward)
            batch_obs_tp1.append(np.array(obs_tp1, copy=False))
            batch_done.append(done)
            idxes.append(idx)

        return np.array(batch_obs_t), np.array(batch_action), np.array(batch_reward), np.array(batch_obs_tp1), \
               np.array(batch_done), idxes

    def get_trajectory(self, idx):
        X = []
        U = []
        at_root = False
        obs_f = None
        while not at_root:
            obs_t, action, _, obs_tp1, _ = self._buffer[idx]
            idx = self.parent_idxes[idx]
            if obs_f is None:
                obs_f = np.expand_dims(obs_tp1, axis=0)
            X.append(obs_t)
            U.append(action)
            if idx == -1:
                at_root = True

        X = np.array(X[::-1])
        U = np.array(U[::-1])

        T = len(U)
        dX = X[0].shape[0]
        dU = U[0].shape[0]

        trajectory = Trajectory(T, dX, dU)
        trajectory.add(X, U)
        trajectory.add(obs_f)

        return trajectory

    def get_policy(self, idx):
        return self.policies[idx]

    def get_obs(self):
        return np.array([np.array(_[0]) for _ in self._buffer])

    def kd_estimate(self, sample_size):
        obs = np.array([np.array(_[0]) for _ in self._buffer])
        kd = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth, leaf_size=self.leaf_size)
        kd.fit(obs)

        samples = self.sample(sample_size)

        scores = np.exp(kd.score_samples(samples[0]))

        return samples, scores

    def get_rgb_array(self, env):
        obs = np.array([np.array(_[0]) for _ in self._buffer])[:, :2]
        kd = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth, leaf_size=self.leaf_size)
        kd.fit(obs)

        x_dim, y_dim = (env._world_size * env.PPM / 5)
        x_dim, y_dim = int(x_dim), int(y_dim)
        img = np.zeros((x_dim, y_dim))
        poses = []
        for i in range(x_dim):
            for j in range(x_dim):
                pos = np.array([i * 2. / x_dim - 1., j * 2. / y_dim - 1.])
                poses.append(pos)
        poses = np.array(poses)

        log_scores = kd.score_samples(poses)
        scores = np.exp(log_scores)
        count = 0
        for i in range(x_dim):
            for j in range(y_dim):
                img[i, j] = scores[count]
                count += 1

        img = img.repeat(5, axis=0).repeat(5, axis=1)
        img -= np.min(img)
        img /= np.max(img)

        cmap = plt.get_cmap('afmhot')
        rgba_img = cmap(img)
        rgb_img = rgba_img[:, :, :3]
        rgb_img *= 255

        return rgb_img


class PrioritizedReplayBuffer(BLPrioritizedReplayBuffer):

    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size, alpha)

    def size(self):
        return len(self._storage())