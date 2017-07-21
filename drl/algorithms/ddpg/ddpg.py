import os
import time
import tensorflow as tf
import numpy as np

from .critic import CriticNetwork
from .actor import ActorNetwork
from drl.utilities.statistics import Statistics, get_summary_dir
from drl.utilities.utilities import print_dict
from drl.replaybuffer import PrioritizedReplayBuffer, ReplayBuffer
from drl.utilities.logger import Logger


class DDPG(object):
    name = 'DDPG'
    tags = ['reward', 'loss', 'max_q']

    """
    Implementation of the Deep Deterministic Policy Gradient algorithm from

        'Continuous Control with Deep Reinforcement Learning, Lillicrap T. et al.' - https://arxiv.org/abs/1509.02971
    """

    def __init__(self,
                 env,
                 exploration_strategy=None,
                 exploration_decay=None,
                 lr_actor=0.0001,
                 lr_critic=0.001,
                 gamma=0.99,
                 tau=0.001,
                 hidden_nodes=[400, 300],
                 batch_norm=False,
                 l2_critic=0.01,
                 scale_reward=1.,
                 replay_buffer_size=1000000,
                 minibatch_size=64,
                 populate_buffer=None,
                 prioritized_replay=False,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 num_episodes=250,
                 max_steps=1000,
                 num_updates_iteration=1,
                 print_info=True,
                 render_env=False,
                 render_freq=25,
                 record=True,
                 record_freq=25,
                 base_dir_summary=None,
                 save=True,
                 save_freq=25,
                 restore=True,
                 restore_checkpoint=None):
        """
        Constructs 'DDPG' object.

        :param env: environment
        :param options_in: available and default options for DDPG object:

            'lr_actor': 0.0001,             # Learning rate actor
            'lr_critic': 0.001,             # Learning rate critic
            'gamma': 0.99,                  # Gamma for Q-learning update
            'tau': 0.001,                   # Soft target update parameter
            'hidden_nodes': [400, 300],     # Number of hidden nodes per layer
            'batch_norm': False,            # True: use batch normalization otherwise False
                                            #       Only observation input is normalized!
            'l2_critic': 0.01,              # L2 regularization term for critic
            'prioritized_replay': False,    # Use prioritized experience replay
            'prioritized_replay_alpha': 0.6,# Amount of prioritization to use (0 - None, 1 - Full)
            'prioritized_replay_beta': 0.4, # Importance weight for prioritized replay buffer (0 - No correction, 1 - Full correction)
            'batch_size': 64,               # Mini-batch size
            'buffer_size': 1000000,         # Size of replay buffer
            'num_updates_iter': 1,          # Number of updates per iteration
        """
        self.env = env
        self.exploration_strategy = exploration_strategy
        self.exploration_decay = exploration_decay
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.hidden_nodes = hidden_nodes
        self.batch_norm = batch_norm
        self.l2_critic = l2_critic
        self.scale_reward = scale_reward
        self.replay_buffer_size = replay_buffer_size
        self.minibatch_size = minibatch_size
        if populate_buffer is None:
            self.populate_buffer = self.minibatch_size
        else:
            self.populate_buffer = populate_buffer
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta = prioritized_replay_beta
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.num_updates_iteration = num_updates_iteration
        self.render_env = render_env
        self.render_freq = render_freq
        self.record = record
        self.record_freq = record_freq
        self.base_dir_summary = base_dir_summary
        self.save = save
        self.save_freq = save_freq
        self.restore = restore
        self.restore_checkpoint = restore_checkpoint

        # Actor and critic arguments
        network_args = {
            'obs_dim': self.env.observation_space.shape[0],
            'action_dim': self.env.action_space.shape[0],
            'tau': tau,
            'hidden_nodes': hidden_nodes,
            'batch_norm': batch_norm
        }

        # Initialize actor and critic network
        self.actor = ActorNetwork(learning_rate=self.lr_actor, action_bounds=self.env.action_space.high,
                                  **network_args)
        self.critic = CriticNetwork(learning_rate=self.lr_critic, l2_param=self.l2_critic, **network_args)

        # Create experience replay buffer
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_size, self.prioritized_replay_alpha)
        else:
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        if print_info:
            print_dict("Hyper-parameters", self.__dict__)
            self.actor.print_summary()
            self.critic.print_summary()

    def train(self, sess):
        # Set session and initialize variables
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        # Initialize target networks
        self.actor.init_target_net(self.sess)
        self.critic.init_target_net(self.sess)

        # Initialize statistics module
        statistics = Statistics(self.env.name, self.name, self.tags, self.base_dir_summary)

        logger = Logger(self.num_episodes, 'Episodes')
        for i_episode in range(self.num_episodes):
            obs = self.env.reset()

            # Summary values
            i_step = 0
            ep_reward = 0.
            ep_loss = 0.
            ep_max_q = 0.
            done = False

            if self.exploration_strategy is not None:
                self.exploration_strategy.reset()

            while not done and (i_step < self.max_steps):
                if (self.render_env and i_episode % self.render_freq == 0) or (i_episode == self.num_episodes - 1) or \
                        (self.record and i_episode % self.record_freq == 0):
                    self.env.render(record=(self.record and (i_episode % self.record_freq == 0 or
                                                             i_episode == self.num_episodes - 1)))

                # Get action and add noise
                action = self.get_action(obs)
                if self.exploration_decay is not None:
                    action += self.exploration_decay.sample()
                elif self.exploration_strategy is not None:
                    action += self.exploration_strategy.sample()

                # Take step
                next_obs, reward, done, _ = self.env.step(action[0])

                # Add experience to replay buffer
                self.replay_buffer.add(np.reshape(obs, [self.env.observation_space.shape[0]]),
                                       np.reshape(action, [self.env.action_space.shape[0]]), reward * self.scale_reward,
                                       np.reshape(next_obs, [self.env.observation_space.shape[0]]), done)

                # Update
                loss, max_q = 0, 0
                if self.replay_buffer.size() > self.populate_buffer:
                    for _ in range(self.num_updates_iteration):
                        loss_update, max_q_update = self.update()
                        loss += loss_update
                        max_q += max_q_update

                # Go to next step
                obs = next_obs

                # Update summary values
                ep_reward += reward
                ep_loss += loss
                ep_max_q += max_q
                i_step += 1

            if self.exploration_decay is not None:
                self.exploration_decay.update()

            if (self.render_env and i_episode % self.render_freq == 0 and self.render_freq != 1) or \
                    (i_episode == self.num_episodes - 1) or (self.record and i_episode % self.record_freq == 0):
                self.env.render(close=True, record=(self.record and (i_episode % self.record_freq == 0 or
                                                                     i_episode == self.num_episodes - 1)))

            # Print and save summary values
            ave_ep_loss = ep_loss / i_step / self.num_updates_iteration
            ave_ep_max_q = ep_max_q / i_step / self.num_updates_iteration

            print_values = dict(zip(['steps'] + self.tags, [i_step, ep_reward, ave_ep_loss, ave_ep_max_q]))
            logger.update(1, **print_values)

            statistics.update_tags(i_episode, self.tags, [ep_reward, ave_ep_loss, ave_ep_max_q])
            statistics.save_episode(i_episode, i_step)
            statistics.write()

            if self.save and i_episode % self.save_freq == 0:
                self.save_model(i_episode)

        if self.save:
            self.save_model()

        logger.update(-1) # Close the progress bar
        logger.write(statistics.get_summary_string(), 'blue')

    def get_action(self, obs):
        """
        Predicts action using actor network.

        :param obs: observation
        :return: action
        """
        return self.actor.predict(self.sess, np.reshape(obs, (1, self.env.observation_space.shape[0])), is_training=False)

    def update(self):
        """
        Execute one update step:

            1) Sample mini-batch from replay buffer

            2) Set target
                y(i) = r(i) + gamma * Q'(s(i+1), mu(s(i+1)))

            3) Update critic by minimizing loss
                L = 1/N * SUM( y(i) - Q(s(i), a(i)) )^2

            4) Update actor using policy gradient:
                Grad_th(J) = 1/N * SUM( Grad_a Q(s(i), mu(s(i)) * Grad_th mu(s(i)) )

            5) Update target networks

        :return: average loss of the critic, average max q value
        """
        # Update prediction
        if self.prioritized_replay:
            *minibatch, w, idxes = self.replay_buffer.sample(self.minibatch_size, self.prioritized_replay_alpha)
        else:
            minibatch = self.replay_buffer.sample(self.minibatch_size)
            idxes = None

        # Sample batch
        obs_t_batch, a_batch, r_batch, obs_tp1_batch, t_batch = minibatch

        # Calculate y target
        next_a_batch = self.actor.predict_target(self.sess, obs_tp1_batch)
        target_q = self.critic.predict_target(self.sess, obs_tp1_batch, next_a_batch)

        y_target = np.zeros((target_q.shape[0], 1))
        for i in range(target_q.shape[0]):
            if t_batch[i]:
                # if state is terminal next Q is zero
                y_target[i] = r_batch[i]
            else:
                y_target[i] = r_batch[i] + self.gamma * target_q[i]

        # Update critic
        loss, q = self.critic.train(self.sess, obs_t_batch, a_batch,
                                    np.reshape(y_target, (self.minibatch_size, 1)))

        # Update priorities
        if self.prioritized_replay:
            td_error = np.abs(q - y_target) + 1e-6
            self.replay_buffer.update_priorities(idxes, td_error)

        # Update actor
        mu_batch = self.actor.predict(self.sess, obs_t_batch)
        action_gradients = self.critic.action_gradients(self.sess, obs_t_batch, mu_batch)
        self.actor.train(self.sess, obs_t_batch, action_gradients[0])

        # Update target networks
        self.actor.update_target_net(self.sess)
        self.critic.update_target_net(self.sess)

        return np.mean(loss), np.max(q)

    def print_summary(self):
        """
        Print summaries of actor and critic networks.
        """
        self.actor.print_summary()
        self.critic.print_summary()

    def save_model(self, checkpoint=None):
        """
        Saves the current Tensorflow variables in the specified path, after saving the location is printed.
        All Tensorflow variables are saved, this means you can even continue training if you want.

        :param path: location to save the model
        """
        # TODO: ADD saving the full information of the experiment
        path = os.path.join(get_summary_dir(), 'model')
        saver = tf.train.Saver()
        saver.save(self.sess, path, global_step=checkpoint)

    def restore(self, path, checkpoint=None):
        """
        Restores the Tensorflow variables saved at the specified path.
        :param path: location of the saved model
        """
        # TODO: Add the rest of restore method so an experiment can be fully restored with all settings from file
        saver = tf.train.Saver()
        path = os.path.join(path, 'model')
        if checkpoint is None:
            saver.restore(self.sess, path)
        else:
            saver.restore(self.sess, path + "-" + checkpoint)
