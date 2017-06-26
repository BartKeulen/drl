from unittest import TestCase
import numpy as np
import pytest
from drl.replaybuffer import ReplayBuffer

## Parameters
size = 1000
batch_size = 10


class TestReplayBuffer(TestCase):

    def setUp(self):
        self.buffer = ReplayBuffer(size)

    def test_add(self):
        for i in range(size*2):
            trans = generate_transition()
            self.buffer.add(*trans,,

        self.assertTrue(True)

    def test_size(self):
        for i in range(size):
            trans = generate_transition()
            self.buffer.add(*trans,,
            self.assertEqual(i+1, self.buffer.size())

        for i in range(size):
            trans = generate_transition()
            self.buffer.add(*trans,,
            self.assertEqual(size, self.buffer.size())

    def test_sample_batch(self):
        s = []
        a = []
        r = []
        t = []
        s2 = []
        for i in range(size*2):
            trans = generate_transition()
            self.buffer.add(*trans,,

            s.append(trans[0])
            a.append(trans[1])
            r.append(trans[2])
            t.append(trans[3])
            s2.append(trans[4])

            s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample(batch_size)

            self.assertTrue(np.all([np.any(s == value) for value in s_batch]))
            self.assertTrue(np.all([np.any(a == value) for value in a_batch]))
            self.assertTrue(np.all([np.any(r == value) for value in r_batch]))
            self.assertTrue(np.all([np.any(t == value) for value in t_batch]))
            self.assertTrue(np.all([np.any(s2 == value) for value in s2_batch]))

    def test_clear(self):
        self.assertEqual(0, self.buffer.size())

        for i in range(10):
            trans = generate_transition()
            self.buffer.add(*trans,,

        self.assertEqual(10, self.buffer.size())

        self.buffer.clear()

        self.assertEqual(0, self.buffer.size())


def generate_transition():
    s = np.random.randn(2)
    a = np.random.randn(1)
    r = np.random.randn(1)
    t = False
    s2 = np.random.randn(2)
    return s, a, r, t, s2