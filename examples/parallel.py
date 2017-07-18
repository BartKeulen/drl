import os
import sys

import gym

from drl.algorithms.ddpg import DDPG
from drl.algorithms.rlagent import RLAgent
from drl.env import Pendulum
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy
from drl.utilities.parallel import *

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import conf_cheetah as conf


def task(params):
    env = gym.make(conf.env_name)

    algo = DDPG(env=env,
                options_in=conf.options_algo)

    noise = OrnSteinUhlenbeckStrategy(action_dim=env.action_space.shape[0])

    agent = RLAgent(env=env,
                    algo=algo,
                    exploration_strategy=noise,
                    options_in=conf.options_agent)

    return agent

param_grid = {
    'run': range(5),
}

Parallel(task, param_grid, 5)