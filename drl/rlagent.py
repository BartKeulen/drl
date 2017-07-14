import os
import shutil

import pyglet
import ffmpy
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from drl.utilities import Statistics

options = {
            'num_episodes': 250,    # Number of episodes
            'max_steps': 1000,      # Maximum number of steps per episode
            'num_exp': 1,           # Number of experiments to run
            'parallel': False,      # Number of parallel threads to use
            'render_env': False,    # True: render environment
            'render_freq': 1,       # Frequency to render (not every episode saves computation time)
            'save_freq': None,      # Frequency to save the model parameters and optional video
            'record': False,        # Save a video recording with the model
            'num_tests': 1,
            'render_tests': True,
            'print': True
        }


class RLAgent(object):
    """
    Class for executing reinforcement learning experiments.

    The 'Agent' object makes it easy to conduct different reinforcement learning experiments with different
    environments, algorithms and exploration methods. See base classes for how to construct them.

    """
    # TODO: Built in method for running multiple experiments with different environments, algorithms and exploration methods
    #       Use tuples to input them in the __init__ function and automatically run them all in run_experiment
    #       Change Statistic class to support this

    def __init__(self, env, algo, exploration, exploration_decay=None, options_in=None, base_dir=None, save=False):
        """
        Constructs 'Agent' object.

        :param env: environment
        :param algo: reinforcement learning algorithm
        :param exploration: exploration method
        :param options_in: available and default options for Agent object:

            'num_episodes': 250,    # Number of episodes
            'max_steps': 200,       # Maximum number of steps per episode
            'num_exp': 1,           # Number of experiments to run
            'render_env': False,    # True: render environment
            'render_freq': 1        # Frequency to render (not every episode saves computation time)
        """
        self._env = env
        self._algo = algo
        self._exploration = exploration
        self._exploration_decay = exploration_decay

        if options_in is not None:
            options.update(options_in)

        self._stat = Statistics(env, algo, self, base_dir=base_dir, save=save, print=options['print'])

    def run_experiment(self, sess):
        """
        Runs multiple training sessions
        """
        for run in range(options['num_exp']):
                sess.run(tf.global_variables_initializer())
                self.train(sess, run)

    def train(self, sess, run, parallel=False):
        """
        Executes the training of the learning agent.

        Results are saved using stat object.
        """
        self.dir = self._stat.reset(run)
        self._algo.reset(sess)

        if self._exploration_decay is not None:
            self._exploration_decay.reset()

        x0 = None
        for i_episode in range(options['num_episodes']):
            # if i_episode > 0:
            #     x0 = self._algo.get_initial_state()
            # if x0 is not None:
            #     obs = self._env.reset(x0[:2])
            # else:
            #     obs = self._env.reset()

            obs = self._env.reset()

            i_step = 0
            done = False
            ep_reward = 0.
            self._stat.ep_reset()
            if self._exploration is not None:
                self._exploration.reset()

            while not done and (i_step < options['max_steps']):
                if options['render_env'] and i_episode % options['render_freq'] == 0:
                    self._env.render()

                # Get action and add noise
                # action = self._exploration.sample()
                action = self._algo.get_action(obs)
                if self._exploration_decay is not None:
                    action += self._exploration_decay.sample()
                elif self._exploration is not None:
                    action += self._exploration.sample()

                # Take step
                next_obs, reward, done, _ = self._env.step(action[0])

                # Update
                update_info = self._algo.update(obs, action, reward, next_obs, done)
                self._stat.update(reward, update_info)

                # Go to next step
                obs = next_obs
                ep_reward += reward
                i_step += 1

            if self._exploration_decay is not None:
                self._exploration_decay.update()

            if options['render_env'] and i_episode % options['render_freq'] == 0 and \
                            options['render_freq'] != 1:
                self._env.render(close=True)

            if parallel:
                print("Thread: %d, Episode: %d, Steps: %d, Reward: %.2f" % (run, i_episode, i_step, ep_reward))

            self._stat.write(i_episode, i_step, ep_reward)

            if options['save_freq'] is not None and i_episode % options['save_freq'] == 0:
                self.save(i_episode)

        if options['save_freq'] is not None:
            self.save()

    def save(self, episode=None):
        """

        :param path:
        :param episode:
        :return:
        """
        if episode is not None:
            path = os.path.join(self.dir, str(episode))
        else:
            path = os.path.join(self.dir, 'final')

        # Save current configuration of algorithm
        self._algo.save_model(path)
        if options['record']:
            # Create temporary directory for .png frames
            rec_tmp = os.path.join(path, 'tmp')
            if not os.path.exists(rec_tmp):
                os.makedirs(rec_tmp)

            # Run inference
            self.test(options['num_tests'], options['max_steps'], options['render_tests'], episode, record_dir=rec_tmp)

            # Intialize ffmpy
            ff = ffmpy.FFmpeg(
                inputs={os.path.join(rec_tmp, '*.png'): '-y -framerate 24 -pattern_type glob'},
                outputs={os.path.join(path, 'video.mp4'): None}
            )
            # Send output of ffmpy to log.txt file in temporary record folder
            # if error check the log file
            ff.run(stdout=open(os.path.join(rec_tmp, 'log.txt'), 'w'), stderr=open(os.path.join(rec_tmp, 'tmp.txt'), 'w'))
            # Remove .png and log.txt file
            shutil.rmtree(rec_tmp)

    def test(self, num_episodes, max_steps, render_env, episode=None, record_dir=None):
        """

        :param num_episodes:
        :param max_steps:
        :param render_env:
        :param episode:
        :param record_dir:
        :return:
        """
        for i_episode in range(num_episodes):
            obs = self._env.reset()

            i_step = 0
            done = False
            ep_reward = 0.

            while (not done) and (i_step < max_steps):
                if render_env:
                    self._env.render()

                # Get action
                action = self._algo.get_action(obs)

                # Take step
                next_obs, reward, done, _ = self._env.step(action[0])

                # Go to next step
                obs = next_obs
                ep_reward += reward
                i_step += 1

                if record_dir is not None:
                    pre_zeros = len(str(max_steps)) - len(str(i_step))
                    path = os.path.join(record_dir, str(i_episode) + '0'*pre_zeros + str(i_step))
                    pyglet.image.get_buffer_manager().get_color_buffer().save(path + '.png')

            display_str = '[TEST] '
            if episode is not None:
                display_str += 'Training episode: {:5d} |'.format(episode)

            display_str += 'Test episode: {:5d} | Steps: {:5d} | Reward: {:5.2f}'.format(i_episode, i_step, ep_reward)
            tqdm.write(display_str)

        if render_env:
            self._env.render(close=True)

    @staticmethod
    def get_info():
        return options