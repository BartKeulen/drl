import numpy as np
from baselines.deepq import ReplayBuffer
from sklearn.neighbors import KernelDensity

import time
from tqdm import tqdm


class ReplayBufferKD(ReplayBuffer):

    def __init__(self, size):
        super(ReplayBufferKD, self).__init__(size)

    def kd_estimate(self, tree_size, samples):
        if tree_size > self.__len__():
            tree_size = self.__len__()
        obses_t = self.sample(tree_size)[0]

        kd = KernelDensity()
        kd.fit(obses_t)

        scores = kd.score_samples(samples)

        return scores
