import numpy as np


class RLAgent(object):
    """
    Class for executing reinforcement learning experiments.

    The 'Agent' object makes it easy to conduct different reinforcement learning experiments with different
    environments, algorithms and exploration methods. See base classes for how to construct them.

    """
    # TODO: Built in method for running multiple experiments with different environments, algorithms and exploration methods
    #       Use tuples to input them in the __init__ function and automatically run them all in run_experiment
    #       Change Statistic class to support this

    def __init__(self, env, algo, exploration, stat, options_in=None):
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
            'num_exp': 5,           # Number of experiments to run
            'render_env': False,    # True: render environment
            'render_freq': 1        # Frequency to render (not every episode saves computation time)
        """
        self._env = env
        self._algo = algo
        self._exploration = exploration
        self._stat = stat

        self.options = {
            'num_episodes': 250,    # Number of episodes
            'max_steps': 200,       # Maximum number of steps per episode
            'num_exp': 5,           # Number of experiments to run
            'render_env': False,    # True: render environment
            'render_freq': 1        # Frequency to render (not every episode saves computation time)
        }
        if options_in is not None:
            self.options.update(options_in)

    def train(self, run=0):
        """
        Executes the training of the learning agent.

        Results are saved using stat object.
        """
        self._stat.reset(run)
        self._algo.reset()

        print('\n------------------ Start training ------------------\n')
        for i_episode in range(self.options['num_episodes']):
            obs = self._env.reset()

            i_step = 0
            terminal = False
            ep_reward = 0.
            self._stat.episode_reset()
            self._exploration.reset()

            while (not terminal) and (i_step < self.options['max_steps']):
                if self.options['render_env'] and i_episode % self.options['render_freq'] == 0:
                    self._env.render()

                # Get action and add noise
                action = self._algo.get_action(obs) + self._exploration.sample()

                # Take step
                next_obs, reward, terminal, _ = self._env.step(action[0])

                # Update
                update_info = self._algo.update(self._replay_buffer)
                self._stat.update(update_info)

                # Go to next iter
                obs = next_obs
                ep_reward += reward
                i_step += 1

            if self.options['render_env'] and i_episode % self.options['render_freq'] == 0 and \
                            self.options['render_freq'] != 1:
                self._env.render(close=True)

            self._stat.write(ep_reward, i_episode, i_step)

        print('\n------------------  End training  ------------------\n')

    def run_experiment(self):
        """
        Runs multiple training sessions
        """
        for run in range(self.options['num_exp']):
            self.train(run)
