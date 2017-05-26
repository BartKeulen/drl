import tensorflow as tf
from deepreinforcementlearning.replaybuffer import ReplayBuffer
from network import NAFNetwork
import numpy as np

SUMMARY_TAGS = ['q', 'v', 'a', 'mu', 'loss']


class NAF(object):

    def __init__(self,
                 sess,
                 env,
                 stat,
                 learning_rate,
                 gamma,
                 tau,
                 num_updates_iter,
                 exploration,
                 buffer_size,
                 batch_size):
        self.sess = sess
        self.env = env
        self.stat = stat

        self.gamma = gamma
        self.update_repeat = num_updates_iter
        self.exploration = exploration
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.obs_bounds = env.observation_space.high
        self.action_bounds = env.action_space.high

        network_args = {
            'sess': self.sess,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'action_bounds': self.action_bounds,
            'learning_rate': learning_rate
        }

        self.predict_network = NAFNetwork(scope="predict_network", **network_args)
        self.target_network = NAFNetwork(scope="target_network", **network_args)
        self.target_network.make_soft_update_ops(self.predict_network, tau)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def train(self, num_episodes, max_steps, render_env=False):
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

        # Initialize target network
        self.target_network.hard_copy_from(self.predict_network)

        for i_episode in xrange(num_episodes):
            obs = self.env.reset()

            i_step = 0
            terminal = False
            ep_reward = 0.
            self.stat.reset()

            while (not terminal) and (i_step < max_steps):
                if render_env:
                    self.env.render()

                # Get action and add noise
                mu = self.predict_network.predict_mu(np.reshape(obs, (1, self.obs_dim)))

                action = mu + self.exploration.get_noise()

                # Take step
                next_obs, reward, terminal, info = self.env.step(action[0])

                # Add experience to replay buffer
                self.replay_buffer.add(np.reshape(obs, self.obs_dim),
                                       np.reshape(action, self.action_dim), reward, terminal,
                                       np.reshape(next_obs, self.obs_dim))

                # Update
                if self.replay_buffer.size() >= self.batch_size:
                    self.update()

                # Go to next state
                obs = next_obs
                ep_reward += reward
                i_step += 1

            self.stat.write(ep_reward, i_episode, i_step)
            self.exploration.increase()

    def update(self):
        for i_update in xrange(self.update_repeat):
            # Sample batch
            obs_batch, a_batch, r_batch, t_batch, next_obs_batch = \
                self.replay_buffer.sample_batch(self.batch_size)

            # Calculate targets
            target_v = self.target_network.predict_v(next_obs_batch)
            y_target = []
            for i in xrange(target_v.shape[0]):
                if t_batch[i]:
                    y_target.append(r_batch[i])
                else:
                    y_target.append(r_batch[i] + self.gamma * target_v[i])

            # Update network
            q, v, a, mu, loss = self.predict_network.train(obs_batch, a_batch, np.reshape(y_target, (obs_batch.shape[0], 1)))

            self.stat.update({
                'q': np.mean(q),
                'v': np.mean(v),
                'a': np.mean(a),
                'mu': np.mean(mu),
                'loss': np.mean(loss)
            })

        self.target_network.do_soft_update()

    @staticmethod
    def get_summary_tags():
        return SUMMARY_TAGS
