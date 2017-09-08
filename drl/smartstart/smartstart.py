import numpy as np
import tensorflow as tf

from drl.algorithms import DDPG
from drl.replaybuffer import ReplayBufferKD
from drl.utilities import Statistics
from drl.utilities.logger import Logger


class SmartStartDDPG(DDPG):

    def __init__(self,
                 prob_smart_start=0.7,
                 kernel='gaussian',
                 bandwidth=0.15,
                 leaf_size=100,
                 sample_size=100,
                 render_traj=False,
                 *args,
                 **kwargs):
        super(SmartStartDDPG, self).__init__(*args, **kwargs)
        self.prob_smart_start = prob_smart_start
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.leaf_size = leaf_size
        self.sample_size = sample_size
        self.render_traj = render_traj
        self.replay_buffer = ReplayBufferKD(self.replay_buffer_size, kernel, bandwidth, leaf_size)

    def train(self, sess, logger_prefix=None):
        # Set session and initialize variables
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        # Initialize target networks
        self.actor.init_target_net(self.sess)
        self.critic.init_target_net(self.sess)

        # Initialize statistics module
        statistics = Statistics(self.env.name, self.name, self.tags, self.base_dir_summary)

        self.logger = Logger(self.num_episodes, logger_prefix)
        for i_episode in range(self.num_episodes):
            # Summary values
            i_step = 0
            ep_reward = 0.
            ep_loss = 0.
            ep_max_q = 0.
            done = False

            self.replay_buffer.new_episode()

            obs = self.env.reset()
            if i_episode > 0 and np.random.rand(1) < self.prob_smart_start:
                obs, ep_reward, done, _ = self.smart_start(render=(i_episode % self.render_freq == 0))

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

            if self.render_traj:
                traj = self.replay_buffer.get_trajectory(self.replay_buffer.parent)
                self.env.add_trajectory(traj, (255, 255, 255, 255))

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
            self.logger.update(1, **print_values)

            statistics.update_tags(i_episode, self.tags, [ep_reward, ave_ep_loss, ave_ep_max_q])
            statistics.save_episode(i_episode, i_step)
            statistics.write()

            if self.save and i_episode % self.save_freq == 0:
                self.save_model(i_episode)

        if self.save:
            self.save_model()

        self.logger.update(-1)  # Close the progress bar
        self.logger.write(statistics.get_summary_string(), 'blue')

    def smart_start(self, render):
        samples, scores = self.replay_buffer.kd_estimate(self.sample_size)
        argmin_score = np.argmin(scores) # TODO: implement use softmax and select with probability instead of minimum

        start_idx = self.replay_buffer.parent_idxes[samples[-1][argmin_score]]
        traj = self.replay_buffer.get_traj_sample(start_idx)
        return self.replay_trajectory(traj, render)

    def sample_buffer(self):
        *minibatch, idxes = self.replay_buffer.sample(self.minibatch_size)
        return minibatch, None, idxes

    def replay_trajectory(self, traj, render=False, add_to_buffer=False):
        obs_t = self.env.reset()
        U = traj.get_U()
        step = 0
        reward = 0.
        done = False
        while not done and step < traj.T:
            obs_tp1, r, t, _ = self.env.step(U[step, :])

            reward += r

            if add_to_buffer:
                self.replay_buffer.add(obs_t, U[step, :], r, obs_tp1, t)

            obs_t = obs_tp1
            step += 1

            if t:
                done = True

            if render:
                self.env.render()

        if self.render_traj:
            traj = self.replay_buffer.get_trajectory(self.replay_buffer.parent)
            self.env.add_trajectory(traj, (255, 0, 0, 255))

        return obs_t, reward, done, step