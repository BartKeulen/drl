from unittest import TestCase
import tensorflow as tf
import numpy as np
from src.statistics import Statistics


def simulate_training(stat):
    summary_updates = {}

    for j in xrange(5):
        for i in xrange(20):
            for tag in ['reward', 'q-value']:
                summary_updates[tag] = np.random.rand(1)[0]

            if i == 19:
                terminal = True
            else:
                terminal = False

            stat.update(summary_updates, terminal, j, i)


class TestStatistics(TestCase):

    with tf.Session() as sess:
        stat = Statistics(sess, 'pendulum', 'random', 1, ['reward', 'q-value'])

        simulate_training(stat)

        stat.reset()

        simulate_training(stat)
