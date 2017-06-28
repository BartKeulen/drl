import os, shutil

import numpy as np
import pyglet
import ffmpy
from tqdm import tqdm

from drl.utilities import Statistics

ndash = '-' * 50
options = {
            'num_episodes': 250,    # Number of episodes
            'max_steps': 1000,       # Maximum number of steps per episode
            'num_exp': 1,           # Number of experiments to run
            'render_env': False,    # True: render environment
            'render_freq': 1,       # Frequency to render (not every episode saves computation time)
            'save_freq': None,      # Frequency to save the model parameters and optional video
            'record': False         # Save a video recording with the model
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

    def __init__(self, env, algo, exploration, options_in=None):
        """
        Constructs 'Agent' object.

        :param env: environment
        :param algo: reinforcement learning algorithm
        :param exploration: exploration method
        :param replay_buffer: replay buffer
        :param stat: statistics object
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

        if options_in is not None:
            options.update(options_in)

        self._stat = Statistics(env, algo, self)

    def run_experiment(self):
        """
        Runs multiple training sessions
        """
        for run in range(options['num_exp']):
            self.dir = self._stat.reset(run)
            self._algo.reset()
            self.train()

    def train(self):
        """
        Executes the training of the learning agent.

        Results are saved using stat object.
        """
        for i_episode in range(options['num_episodes']):
            # init_obs = self._algo.get_initial_state()
            # if init_obs is not None:
            #     init_state = np.array([np.arccos(init_obs[0]), 0.])
            # else:
            #     init_state = init_obs
            # obs = self._env.reset(state=init_state)
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
                action = self._algo.get_action(obs)
                if self._exploration is not None:
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

            if options['render_env'] and i_episode % options['render_freq'] == 0 and \
                            options['render_freq'] != 1:
                self._env.render(close=True)

            self._stat.write(i_episode, i_step, ep_reward)

            if options['save_freq'] is not None and i_episode % options['save_freq'] == 0:
                self.save(i_episode)

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

        print('')
        # Save current configuration of algorithm
        self._algo.save_model(path)
        if options['record']:
            # Create temporary directory for .png frames
            rec_tmp = os.path.join(path, 'tmp')
            if not os.path.exists(rec_tmp):
                os.makedirs(rec_tmp)

            # Run inference
            self.test(1, options['max_steps'], True, rec_tmp)

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
        print('')

    def test(self, num_episodes, max_steps, render_env, record_dir=None):
        """

        :param num_episodes:
        :param max_steps:
        :param render_env:
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

            print('[TEST] Episode: {:5d} | Steps: {:5d} | Reward: {:5.2f}'.format(i_episode, i_step, ep_reward))

        if render_env:
            self._env.render(close=True)

    @staticmethod
    def restore(path):
        pass

    @staticmethod
    def get_info():
        return options