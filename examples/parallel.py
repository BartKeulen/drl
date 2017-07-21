import time

import gym
import tensorflow as tf
from drl.algorithms.ddpg import DDPG
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy
from drl.utilities.scheduler import LinearScheduler
from drl.utilities.experimenter import run_experiment, Mode
from drl.env.environment import GymEnv

env_name = 'Pendulum-v0'


def task(params):
    env = GymEnv(env_name)

    exploration_strategy = OrnSteinUhlenbeckStrategy(action_dim=env.action_space.shape[0])
    exploration_decay = LinearScheduler(exploration_strategy, start=100, end=125)

    ddpg = DDPG(env=env,
                exploration_strategy=exploration_strategy,
                exploration_decay=exploration_decay,
                record=False,
                print_info=False)

    return ddpg


param_grid = {'task': task, 'num_exp': 5}

run_experiment(param_grid, n_processes=5, mode=Mode.REMOTE)

end = time.time()