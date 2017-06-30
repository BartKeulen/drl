import os
import tensorflow as tf
import numpy as np

from .critic import CriticNetwork
from .actor import ActorNetwork
from drl.replaybuffer import ReplayBuffer

# Algorithm info
info = {
    'name': 'DDPG',
    'summary_tags': ['loss', 'max_Q']
}

# Algorithm options
options = {
    'lr_actor': 0.0001,             # Learning rate actor
    'lr_critic': 0.001,             # Learning rate critic
    'gamma': 0.99,                  # Gamma for Q-learning update
    'tau': 0.001,                   # Soft target update parameter
    'hidden_nodes': [400, 300],     # Number of hidden nodes per layer
    'batch_norm': False,            # True: use batch normalization otherwise False
                                    #       Only observation input is normalized!
    'l2_critic': 0.01,              # L2 regularization term for critic
    'batch_size': 64,               # Mini-batch size
    'buffer_size': 1000000,         # Size of replay buffer
    'num_updates_iter': 1,          # Number of updates per iteration
}


class DDPG(object):
    """
    Implementation of the Deep Deterministic Policy Gradient algorithm from

        'Continuous Control with Deep Reinforcement Learning, Lillicrap T. et al.' - https://arxiv.org/abs/1509.02971
    """

    def __init__(self,
                 env,
                 options_in=None):
        """
        Constructs 'DDPG' object.

        :param sess: Tensorflow session
        :param env: environment
        :param options_in: available and default options for DDPG object:

            'lr_actor': 0.0001,             # Learning rate actor
            'lr_critic': 0.001,             # Learning rate critic
            'gamma': 0.99,                  # Gamma for Q-learning update
            'l2_actor': 0.,                 # L2 regularization term for actor
            'l2_critic': 0.01,              # L2 regularization term for critic
            'hidden_nodes': [400, 300],     # Number of hidden nodes per layer
            'batch_norm': False,            # True: use batch normalization otherwise False.
                                            #       Only observation input is normalized!
            'batch_size': 64,               # Mini-batch size
            'buffer_size': 1000000,         # Size of replay buffer
            'num_updates_iter': 1,          # Number of updates per iteration
        """
        self._env = env
        self._replay_buffer = ReplayBuffer(options['buffer_size'])

        # Update options
        if options_in is not None:
            options.update(options_in)

        # Actor and critic arguments
        network_args = {
            'obs_dim': self._env.observation_space.shape[0],
            'action_dim': self._env.action_space.shape[0],
            'tau': options['tau'],
            'hidden_nodes': options['hidden_nodes'],
            'batch_norm': options['batch_norm']
        }

        # Initialize actor and critic network
        self.actor = ActorNetwork(learning_rate=options['lr_actor'], action_bounds=self._env.action_space.high,
                                  **network_args)
        self.critic = CriticNetwork(learning_rate=options['lr_critic'], l2_param=options['l2_critic'], **network_args)

    def reset(self, sess):
        """
        Resets the algorithm and re-initializes all the variables
        """
        self._sess = sess
        self.actor.init_target_net(self._sess)
        self.critic.init_target_net(self._sess)

    def get_action(self, obs):
        """
        Predicts action using actor network.

        :param obs: observation
        :return: action
        """
        return self.actor.predict(self._sess, np.reshape(obs, (1, self._env.observation_space.shape[0])), phase=False)

    def update(self, obs, action, reward, next_obs, done):
        """
        First the latest transition is added to the replay buffer.
        Then performs the update, 'num_updater_iter' times a mini-batch is sampled for updating the actor and critic.
        Afterwards the target networks are updated.

        :param replay_buffer: replay buffer
        :return: average loss of the critic
        """
        # Add experience to replay buffer
        self._replay_buffer.add(np.reshape(obs, [self._env.observation_space.shape[0]]),
                                np.reshape(action, [self._env.action_space.shape[0]]), reward,
                                np.reshape(next_obs, [self._env.observation_space.shape[0]]), done)

        # If not enough samples in replay buffer return
        if self._replay_buffer.size() < options['batch_size']:
            return {'loss': 0., 'max_Q': 0.}

        # Update prediction networks
        loss = 0.
        q = 0.
        for _ in range(options['num_updates_iter']):
            minibatch = self._replay_buffer.sample(options['batch_size'])
            l, q_up = self._update_predict(minibatch)
            loss += l
            q += q_up

        # Update target networks
        self._update_target()

        return {'loss': loss/options['num_updates_iter'],
                'max_Q': q/options['num_updates_iter']}

    def _update_predict(self, minibatch):
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
        obs_batch, a_batch, r_batch, t_batch, next_obs_batch = minibatch

        # Calculate y target
        next_a_batch = self.actor.predict_target(self._sess, next_obs_batch)
        target_q = self.critic.predict_target(self._sess, next_obs_batch, next_a_batch)
        y_target = []

        for i in range(target_q.shape[0]):
            if t_batch[i]:
                # if state is terminal next Q is zero
                y_target.append(r_batch[i])
            else:
                y_target.append(r_batch[i] + options['gamma'] * target_q[i])

        # Update critic
        loss, q = self.critic.train(self._sess, obs_batch, a_batch, np.reshape(y_target, (options['batch_size'], 1)))

        # Update actor
        mu_batch = self.actor.predict(self._sess, obs_batch)
        action_gradients = self.critic.action_gradients(self._sess, obs_batch, mu_batch)
        self.actor.train(self._sess, obs_batch, action_gradients[0])

        return np.mean(loss), np.max(q)

    def _update_target(self):
        """
        Updates target networks for actor and critic.
        """
        self.actor.update_target_net(self._sess)
        self.critic.update_target_net(self._sess)

    def print_summary(self):
        """
        Print summaries of actor and critic networks.
        """
        self.actor.print_summary()
        self.critic.print_summary()

    def save_model(self, path):
        """
        Saves the current Tensorflow variables in the specified path, after saving the location is printed.
        All Tensorflow variables are saved, this means you can even continue training if you want.

        :param path: location to save the model
        """
        path = os.path.join(path, 'model')
        saver = tf.train.Saver()
        saver.save(self._sess, path)

    def restore_model(self, path):
        """
        Restores the Tensorflow variables saved at the specified path.
        :param path: location of the saved model
        """
        saver = tf.train.Saver()
        saver.restore(self._sess, path)

    @staticmethod
    def get_info():
        """
        :return: algorithm info, options
        """
        return info, options
