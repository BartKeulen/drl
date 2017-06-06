from .critic import CriticNetwork
from .actor import ActorNetwork
from drl.replaybuffer import ReplayBuffer, ReplayBufferTF
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from drl.rrtexploration import Trajectory


SUMMARY_TAGS = ['q', 'loss', 'mu', 'r_int']


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
            'hidden_nodes': hidden_nodes,
            'batch_norm': batch_norm
        }

        # Initialize actor
        self.predict_actor = ActorNetwork(scope="predict_actor", learning_rate=learning_rate_actor,
                                          action_bounds=self.action_bounds, **network_args)
        self.target_actor = ActorNetwork(scope="target_actor", learning_rate=learning_rate_actor,
                                         action_bounds=self.action_bounds, **network_args)
        self.target_actor.make_soft_update_ops(self.predict_actor, tau)

        # Initialize critic
        self.predict_critic = CriticNetwork(scope="predict_critic", learning_rate=learning_rate_critic, **network_args)
        self.target_critic = CriticNetwork(scope="target_critic", learning_rate=learning_rate_critic, **network_args)
        self.target_critic.make_soft_update_ops(self.predict_critic, tau)

        # Initialize replay buffer
        # self.replay_buffer = ReplayBuffer(buffer_size)
        self.replay_buffer = ReplayBufferTF(self.sess, self.obs_dim, self.obs_bounds, 25., buffer_size)

    def train(self, num_episodes, max_steps, render_env=False):
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

        # Initialize target networks
        self.target_actor.hard_copy_from(self.predict_actor)
        self.target_critic.hard_copy_from(self.predict_critic)

        # self.env.toggle_plot_density()
        # self.env.toggle_plot_trajectories()

        trajectory_list = []

        for i_episode in xrange(num_episodes):
            obs = self.env.reset()

            i_step = 0
            terminal = False
            ep_reward = 0.
            self.stat.reset()
            self.exploration.reset()

            cur_trajectory = Trajectory()

            while (not terminal) and (i_step < max_steps):
                if render_env:
                    self.env.render()

                # Get action and add noise
                action = self.predict_actor.predict(np.reshape(obs, (1, self.obs_dim))) + self.exploration.sample()
                # action = self.exploration.get_noise()

                cur_trajectory.add_node(obs, action)

                next_obs, reward, terminal, info = self.env.step(action[0])

                self.replay_buffer.add(np.reshape(obs, self.obs_dim),
                              np.reshape(action, self.action_dim), reward, terminal,
                              np.reshape(next_obs, self.obs_dim))

                if self.replay_buffer.size() > self.batch_size:
                    for _ in range(self.num_updates_iter):
                        self.update()

                obs = next_obs
                ep_reward += reward
                i_step += 1

            self.stat.write(ep_reward, i_episode, i_step)
            self.exploration.next_episode()

            trajectory_list.append(cur_trajectory)
            # self.plot_density()
            # self.env.add_trajectory(cur_trajectory)

    def update(self):
        # Sample batch
        obs_batch, a_batch, r_batch, t_batch, next_obs_batch = \
            self.replay_buffer.sample_batch(self.batch_size)

        # Calculate targets
        next_a_batch = self.target_actor.predict(next_obs_batch)
        target_q = self.target_critic.predict(next_obs_batch, next_a_batch)
        y_target = []

        all_r_int = []
        for i in xrange(target_q.shape[0]):
            r_int = 0.
            # r_int = -self.replay_buffer.calc_density(np.reshape(obs_batch[i], [1, 2]))*10.
            all_r_int.append(r_int)

            if t_batch[i]:
                y_target.append(r_batch[i] + r_int)
            else:
                y_target.append(r_batch[i] + r_int + self.gamma * target_q[i])

        # update networks
        q, loss = self.predict_critic.train(obs_batch, a_batch, np.reshape(y_target, (self.batch_size, 1)))

        mu_batch = self.predict_actor.predict(obs_batch)
        action_gradients = self.predict_critic.action_gradients(obs_batch, mu_batch)
        mu = self.predict_actor.train(obs_batch, action_gradients[0])

        # update target networks
        self.target_actor.do_soft_update()
        self.target_critic.do_soft_update()

        # Update stats
        self.stat._update({
            'q': np.mean(q),
            'loss': np.mean(loss),
            'mu': np.mean(mu),
            'r_int': np.mean(all_r_int)
        })

    @staticmethod
    def get_summary_tags():
        return SUMMARY_TAGS

    def plot_density(self):
        w, h = self.env.w, self.env.h

        x = np.linspace(-self.obs_bounds[0], self.obs_bounds[0], w)
        y = np.linspace(-self.obs_bounds[1], self.obs_bounds[1], h)

        xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

        values = np.zeros((w, h))
        for i in range(w):
            for j in range(h):
                values[i, j] = self.replay_buffer.calc_density(np.array([[xv[i, j], yv[i, j]]]))

        self.env.update_density_map(values)
