import time

import gym
import tensorflow as tf
from drl.algorithms.ddpg import DDPG
from drl.algorithms.rlagent import RLAgent
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy
from drl.utilities.scheduler import LinearScheduler
from drl.utilities.experimenter import run_experiment, Mode

env_name = 'Pendulum-v0'

options_ddpg = {
    'batch_norm': False,
    'l2_critic': 0.01,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300]
}

options_agent = {
    'render_env': False,
    'num_episodes': 200,
    'max_steps': 200,
    'num_exp': 5
}

start = time.time()

def task(params):
    env = gym.make(env_name)

    ddpg = DDPG(env=env,
                options_in=options_ddpg)

    exploration_strategy = OrnSteinUhlenbeckStrategy(action_dim=env.action_space.shape[0])
    exploration_decay = LinearScheduler(exploration_strategy, start=100, end=125)

    agent = RLAgent(env=env,
                    algo=ddpg,
                    exploration_strategy=exploration_strategy,
                    exploration_decay=exploration_decay,
                    options_in=options_agent)

    return agent


param_grid = {'task': task, 'num_exp': 5}

run_experiment(param_grid, n_processes=5, mode=Mode.REMOTE)

end = time.time()

print('Time elapsed: %.2f s' % (end - start))