from .critic import CriticNetwork
from .actor import ActorNetwork
from drl.replaybuffer import ReplayBuffer
import tensorflow as tf
import numpy as np

# Summary tags for values to be displayed in tensorboard
SUMMARY_TAGS = ['loss']


class DDPG(object):
    """
    Implementation of the Deep Deterministic Policy Gradient algorithm from

        'Continuous Control with Deep Reinforcement Learning, Lillicrap T. et al.' - https://arxiv.org/abs/1509.02971
    """

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
        """
        Constructs 'DDPG' object.

        :param sess: tensorflow session
        :param env: environment
        :param stat: 'Statistic' object
        :param learning_rate_actor:
        :param learning_rate_critic:
        :param gamma:
        :param tau:
        :param hidden_nodes: array with each entry the number of hidden nodes in that layer.
                                Length of array is the number of hidden layers.
        :param batch_norm: True: use batch normalization otherwise False
        :param exploration: 'Exploration' object
        :param num_updates_iter: number of updates per step
        :param buffer_size: size of the replay buffer
        :param batch_size: mini-batch size
        """
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

        # Initialize actor and critic network
        self.actor = ActorNetwork(learning_rate=learning_rate_actor, action_bounds=self.action_bounds, **network_args)
        self.critic = CriticNetwork(learning_rate=learning_rate_critic, **network_args)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def train(self, num_episodes, max_steps, render_env=False):
        """
        Executes the training of the DDPG agent.

        Results are saved using stat object.

        :param num_episodes: number of episodes
        :param max_steps: maximum number of steps per episode
        :param render_env: True: render environment otherwise False
        """

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

        print('\n------------------ Start training ------------------\n')
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
                action = self._get_action(obs) + self.exploration.sample()

                # Take step
                next_obs, reward, terminal, info = self.env.step(action[0])

                # Add experience to replay buffer
                self.replay_buffer.add(np.reshape(obs, self.obs_dim),
                              np.reshape(action, self.action_dim), reward, terminal,
                              np.reshape(next_obs, self.obs_dim))

                # If enough experiences update #num_updates_iter
                if self.replay_buffer.size() > self.batch_size:
                    for _ in xrange(self.num_updates_iter):
                        self._update()

                # update target networks
                self._update_target()

                # Go to next iter
                obs = next_obs
                ep_reward += reward
                i_step += 1

            self.stat.write(ep_reward, i_episode, i_step)
            self.exploration.next_episode()

        print('\n------------------  End training  ------------------\n')

    def _get_action(self, obs):
        """
        Predicts action using actor network.

        :param obs: observation
        :return: action
        """
        return self.actor.predict(np.reshape(obs, (1, self.obs_dim)))

    def _update(self):
        """
        Executes one update step:

            1) Sample mini-batch from replay buffer

            2) Set target
                y(i) = r(i) + gamma * Q'(s(i+1), mu(s(i+1)))

            3) Update critic by minimizing loss
                L = 1/N * SUM( y(i) - Q(s(i), a(i)) )^2

            4) Update actor using policy gradient:
                Grad_th(J) = 1/N * SUM( Grad_a Q(s(i), mu(s(i)) * Grad_th mu(s(i)) )
        """
        # Sample batch
        obs_batch, a_batch, r_batch, t_batch, next_obs_batch = \
            self.replay_buffer.sample_batch(self.batch_size)

        # Calculate y target
        next_a_batch = self.actor.predict_target(next_obs_batch)
        target_q = self.critic.predict_target(next_obs_batch, next_a_batch)
        y_target = []

        for i in xrange(target_q.shape[0]):
            if t_batch[i]:
                # if state is terminal next Q is zero
                y_target.append(r_batch[i])
            else:
                y_target.append(r_batch[i] + self.gamma * target_q[i])

        # Update critic
        loss = self.critic.train(obs_batch, a_batch, np.reshape(y_target, (self.batch_size, 1)))

        # Update actor
        mu_batch = self.actor.predict(obs_batch)
        action_gradients = self.critic.action_gradients(obs_batch, mu_batch)
        self.actor.train(obs_batch, action_gradients[0])

        # Update statistics
        self.stat.update({
            'loss': np.mean(loss)
        })

    def _update_target(self):
        """
        Updates target networks for actor and critic.
        """
        self.actor.update_target_net()
        self.critic.update_target_net()

    @staticmethod
    def get_summary_tags():
        """
        Summary tags are displayed in tensorboard. This function is used by 'Statistics' object for initialization.
        :return: summary_tags
        """
        return SUMMARY_TAGS
