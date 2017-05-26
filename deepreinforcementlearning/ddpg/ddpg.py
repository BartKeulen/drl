from critic import CriticNetwork
from actor import ActorNetwork
from deepreinforcementlearning.replaybuffer import ReplayBuffer, ReplayBufferTF
import tensorflow as tf
import keras

import numpy as np
import matplotlib.pyplot as plt
from deepreinforcementlearning.rrtexploration import Trajectory


SUMMARY_TAGS = ['loss']


class DDPG(object):

    def __init__(self,
                 sess,
                 env,
                 stat,
                 learning_rate_actor,
                 learning_rate_critic,
                 gamma,
                 tau,
                 hidden_nodes,
                 batch_norm,
                 exploration,
                 num_updates_iter,
                 buffer_size,
                 batch_size):
        self.sess = sess
        self.env = env
        self.stat = stat
        self.gamma = gamma
        self.exploration = exploration
        self.num_updates_iter = num_updates_iter
        self.batch_size = batch_size

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_bounds = self.env.observation_space.high
        self.action_bounds = self.env.action_space.high

        # Actor and critic arguments
        network_args = {
            'sess': self.sess,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'tau': tau,
            'hidden_nodes': hidden_nodes,
            'batch_norm': batch_norm
        }

        self.actor = ActorNetwork(learning_rate=learning_rate_actor, action_bounds=self.action_bounds, **network_args)
        self.critic = CriticNetwork(learning_rate=learning_rate_critic, **network_args)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def train(self, num_episodes, max_steps, render_env=False):
        self.sess.run(tf.global_variables_initializer())

        for i_episode in xrange(num_episodes):
            obs = self.env.reset()

            i_step = 0
            terminal = False
            ep_reward = 0.
            self.stat.reset()
            self.exploration.reset()

            while (not terminal) and (i_step < max_steps):
                if render_env:
                    self.env.render()

                # Get action and add noise
                action = self.actor.predict(np.reshape(obs, (1, self.obs_dim))) + self.exploration.get_noise()

                # Take step
                next_obs, reward, terminal, info = self.env.step(action[0])

                # Add experience to replay buffer
                self.replay_buffer.add(np.reshape(obs, self.obs_dim),
                              np.reshape(action, self.action_dim), reward, terminal,
                              np.reshape(next_obs, self.obs_dim))

                # If enough experiences update #num_updates_iter
                if self.replay_buffer.size() > self.batch_size:
                    for _ in range(self.num_updates_iter):
                        self.update()

                # Go to next iter
                obs = next_obs
                ep_reward += reward
                i_step += 1

            self.stat.write(ep_reward, i_episode, i_step)
            self.exploration.increase()

    def update(self):
        # Sample batch
        obs_batch, a_batch, r_batch, t_batch, next_obs_batch = \
            self.replay_buffer.sample_batch(self.batch_size)

        # Calculate targets
        next_a_batch = self.actor.predict_target(next_obs_batch)
        target_q = self.critic.predict_target(next_obs_batch, next_a_batch)
        y_target = []

        for i in xrange(target_q.shape[0]):
            if t_batch[i]:
                y_target.append(r_batch[i])
            else:
                y_target.append(r_batch[i] + self.gamma * target_q[i])

        # update networks
        loss = self.critic.train(obs_batch, a_batch, np.reshape(y_target, (self.batch_size, 1)))

        mu_batch = self.actor.predict(obs_batch)
        action_gradients = self.critic.action_gradients(obs_batch, mu_batch)
        self.actor.train(obs_batch, action_gradients[0])

        # update target networks
        self.actor.update_target_net()
        self.critic.update_target_net()

        # Update stats
        self.stat.update({
            'loss': np.mean(loss)
        })

    @staticmethod
    def get_summary_tags():
        return SUMMARY_TAGS
