from unittest import TestCase
from drl.replaybuffer import ReplayBuffer
import numpy as np


class TestReplayBuffer(TestCase):
    def test_add(self):
        self.fail()

    def test_size(self):
        self.fail()

    def test_sample_batch(self):
        replay_buffer = ReplayBuffer(50)

        for i in range(10):
            s = np.random.rand(2)
            a = np.random.rand(2)
            r = np.random.rand(1)
            if (np.random.randint(0, 2) == 0):
                t = False
            else:
                t = True
            s2 = np.random.rand(2)
            replay_buffer.add(s, a, r, t, s2)

        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(5)

        print(t_batch + 0)

        self.fail()

    def test_clear(self):
        self.fail()
