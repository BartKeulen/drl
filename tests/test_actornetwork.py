from unittest import TestCase

import numpy as np
import tensorflow as tf
from drl.algorithms.ddpg import ActorNetwork

obs_dim = 2
action_dim = 2
action_bounds = [5., 10]
learning_rate = 1.
tau = 0.5
l2_param = 0.01
hidden_nodes = [10, 10]
batch_norm = False


class TestActorNetwork(TestCase):

    def setUp(self):
        self.sess = tf.InteractiveSession()
        self.network = ActorNetwork(self.sess, obs_dim, action_dim, action_bounds, learning_rate, tau, l2_param,
                                    hidden_nodes, batch_norm)

        self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()

    def test__build_model(self):
        # Test build with different number of hidden nodes and batch_norm = True
        hidden_nodes2 = [10, 10, 10, 10]
        batch_norm2 = True
        network = ActorNetwork(self.sess, obs_dim, action_dim, action_bounds, learning_rate, tau, l2_param,
                               hidden_nodes2, batch_norm2)

        self.assertTrue(True)

    def test_predict(self):
        for i in range(100):
            obs = np.random.randn(obs_dim).reshape([1, obs_dim])
            action = self.network.predict(obs)
            self.assertEqual((1, action_dim), action.shape)

        for i in range(100):
            obs = np.random.randn(obs_dim*10).reshape([10, obs_dim])
            action = self.network.predict(obs)
            self.assertEqual((10, action_dim), action.shape)

    def test_predict_target(self):
        for i in range(100):
            obs = np.random.randn(obs_dim).reshape([1, obs_dim])
            action = self.network.predict_target(obs)
            self.assertEqual((1, action_dim), action.shape)

        for i in range(100):
            obs = np.random.randn(obs_dim*10).reshape([10, obs_dim])
            action = self.network.predict_target(obs)
            self.assertEqual((10, action_dim), action.shape)

    def test_train(self):
        cur_weights = [w.eval().copy() for w in self.network.weights]
        action_gradients = np.ones([10, action_dim])
        obs = np.random.randn(obs_dim*10).reshape([10, obs_dim])

        self.network.train(obs, action_gradients)
        weights = [w.eval().copy() for w in self.network.weights]

        self.assertFalse(np.all([(np.array_equal(w[0], w[1])) for w in zip(cur_weights, weights)]))

    def test_init_target_net(self):
        weights = [w.eval().copy() for w in self.network.weights]
        target_weights = [w.eval().copy() for w in self.network.target_weights]

        self.assertFalse(np.all([(np.array_equal(w[0], w[1])) for w in zip(weights, target_weights)]))

        self.network.init_target_net()
        target_weights = [w.eval().copy() for w in self.network.target_weights]

        self.assertTrue(np.all([(np.array_equal(w[0], w[1])) for w in zip(weights, target_weights)]))

    def test_update_target_net(self):
        weights = self.network.weights
        target_weights = self.network.target_weights

        for i in range(len(weights)):
            weights[i].assign(np.ones(weights[i].shape)).eval()
            target_weights[i].assign(np.ones(weights[i].shape)*2).eval()

        self.network.update_target_net()

        target_weights = [w.eval().copy() for w in self.network.target_weights]
        self.assertTrue(np.any([(np.array_equal(np.ones(w.shape)*1.5, w)) for w in target_weights]))

    def test_print_summary(self):
        self.network.print_summary()
        self.assertTrue(True)