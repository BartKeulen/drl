from drl.utilities import print_dict

ndash = '-' * 50
options = {
            'num_episodes': 250,    # Number of episodes
            'max_steps': 200,       # Maximum number of steps per episode
            'num_exp': 1,           # Number of experiments to run
            'render_env': False,    # True: render environment
            'render_freq': 1        # Frequency to render (not every episode saves computation time)
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
            'num_exp': 1,           # Number of experiments to run
            'render_env': False,    # True: render environment
            'render_freq': 1        # Frequency to render (not every episode saves computation time)
        """
        self._env = env
        self._algo = algo
        self._exploration = exploration
        self._stat = stat

        if options_in is not None:
            options.update(options_in)

        print_dict("Agent options:", options)

    def train(self, run=0):
        """
        Executes the training of the learning agent.

        Results are saved using stat object.
        """
        self._stat.reset(run)
        self._algo.reset()

        self._algo.print_summary()

        print('\n\033[1m{:s} Start training {:s}\033[0m\n'.format(ndash, ndash))
        for i_episode in range(options['num_episodes']):
            obs = self._env.reset()

            i_step = 0
            done = False
            ep_reward = 0.
            self._stat.episode_reset()
            self._exploration.reset()

            while (not done) and (i_step < options['max_steps']):
                if options['render_env'] and i_episode % options['render_freq'] == 0:
                    self._env.render()

                # print(obs)

                # Get action and add noise
                action = self._algo.get_action(obs) + self._exploration.sample()

                # Take step
                next_obs, reward, done, _ = self._env.step(action[0])

                # Update
                update_info = self._algo.update(obs, action, reward, done, next_obs)
                self._stat.update(update_info)

                # Go to next iter
                obs = next_obs
                ep_reward += reward
                i_step += 1

            if options['render_env'] and i_episode % options['render_freq'] == 0 and \
                            options['render_freq'] != 1:
                self._env.render(close=True)

            self._stat.write(ep_reward, i_episode, i_step)

        print('\n\033[1m{:s}  End training  {:s}\033[0m\n'.format(ndash, ndash))

    def run_experiment(self):
        """
        Runs multiple training sessions
        """
        for run in range(options['num_exp']):
            self.train(run)