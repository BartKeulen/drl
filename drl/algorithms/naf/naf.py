import tensorflow as tf
from drl.replaybuffer import ReplayBuffer
from .network import NAFNetwork
import numpy as np

SUMMARY_TAGS = ['loss']


class NAF(object):

    def __init__(self,
                 sess,
                 env,
                 stat,
                 learning_rate,
                 gamma,
                 tau,
                 hidden_nodes,
                 batch_norm,
                 num_updates_iter,
                 exploration,
                 buffer_size,
                 batch_size,
                 seperate_networks):
        self.sess = sess
        self.env = env
        self.stat = stat
        self.gamma = gamma
        self.exploration = exploration
        self.buffer_size = buffer_size
        self.num_updates_iter = num_updates_iter
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
            'learning_rate': learning_rate,
            'tau': tau,
            'hidden_nodes': hidden_nodes,
            'batch_norm': batch_norm,
            'seperate_networks': seperate_networks
        }

        self.network = NAFNetwork(**network_args)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def train(self, num_episodes, max_steps, render_env=False):
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

        # Initialize target network
        self.network.init_target_net()

        for i_episode in range(num_episodes):
            obs = self.env.reset()

            i_step = 0
            terminal = False
            ep_reward = 0.
            self.stat.episode_reset()
            self.exploration.reset()

            while (not terminal) and (i_step < max_steps):
                if render_env:
                    self.env.render()

                # Get action and add noise
                action = self._get_action(obs) + self.exploration.sample()

                # Take step
                next_obs, reward, terminal, _ = self.env.step(action[0]*self.action_bounds)

                # Add experience to replay buffer
                self.replay_buffer.add(np.reshape(obs, self.obs_dim), np.reshape(action, self.action_dim), reward,
                                       np.reshape(next_obs, self.obs_dim), terminal)

                # Update
                if self.replay_buffer.size() >= self.batch_size:
                    for _ in range(self.num_updates_iter):
                        self._update()

                # update target networks
                self._update_target()

                # Go to next state
                obs = next_obs
                ep_reward += reward
                i_step += 1

            self.stat.write(ep_reward, i_episode, i_step)
            self.exploration.next_episode()

    def _get_action(self, obs):
        return self.network.predict_mu(np.reshape(obs, (1, self.obs_dim)))

    def _update(self):
        # Sample batch
        obs_batch, a_batch, r_batch, t_batch, next_obs_batch = \
            self.replay_buffer.sample(self.batch_size)

        # Calculate targets
        target_v = self.network.predict_target_v(next_obs_batch)
        y_target = []
        for i in range(target_v.shape[0]):
            if t_batch[i]:
                y_target.append(r_batch[i])
            else:
                y_target.append(r_batch[i] + self.gamma * target_v[i])

        # Update network
        loss = self.network.train(obs_batch, a_batch, np.reshape(y_target, (obs_batch.shape[0], 1)))

        # Update statistics
        self.stat.update({
            'loss': np.mean(loss)
        })

    def _update_target(self):
        self.network.update_target_net()

    @staticmethod
    def get_summary_tags():
        return SUMMARY_TAGS
