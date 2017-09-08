from drl.algorithms.ddpg import DDPG
from drl.smartstart.smartstart import SmartStartDDPG
from drl.env import Maze
from drl.explorationstrategy import WhiteNoiseStrategy
from drl.utilities.experimenter import run_experiment


def task(params):
    env = Maze.generate_maze(Maze.SIMPLE)

    exploration_strategy = WhiteNoiseStrategy(action_dim=env.action_space.shape[0],
                                          scale=env.action_space.high)

    agent = params['agent'](env=env,
                            exploration_strategy=exploration_strategy,
                            num_episodes=100,
                            max_steps=10000,
                            hidden_nodes=[50, 50],
                            scale_reward=10.,
                            # lr_actor=params['lr'][0],
                            # lr_critic=params['lr'][1],
                            render_env=True,
                            render_freq=10,
                            record=False,
                            print_info=False)

    return agent


# param_grid = {'task': task, 'agent': [DDPG, SmartStartDDPG], 'num_exp': 4}
param_grid = {'task': task, 'agent': [DDPG], 'num_exp': 1}

run_experiment(param_grid, n_processes=1)