from multiprocessing import Pool

import tensorflow as tf
from sklearn.model_selection import ParameterGrid

from drl.utilities.statistics import set_base_dir
from drl.algorithms.algorithm import Algorithm
from drl.utilities.logger import Logger, LogMode


class Mode:
    LOCAL = 'local'
    REMOTE = 'remote'


def process_task(params):
    # TODO: Turn into function decorator (https://www.thecodeship.com/patterns/guide-to-python-function-decorators/)
    print("\033[1mProcess %d started\033[0m" % params['id'])

    graph = tf.Graph()

    with graph.as_default():
        sess = tf.Session(config=tf.ConfigProto(gpu_options=params['gpu_options']))

        agent = params['task'](params)

        if not issubclass(agent.__class__, Algorithm):
            raise Exception('The defined task function has to return a Algorithm object.')

        agent.train(sess)

        sess.close()

        print('\033[1mProcess %d finished\033[0m' % params['id'])


def run_experiment(param_grid, n_processes=1, mode=Mode.LOCAL, base_dir=None):
    """
    Method used for running a reinforcement learning experiment.

    :param param_grid: parameter grid defining tasks and parameter, see explanation above.
    :param n_processes: when running in parallel, the number of parallel processes is defined here
    :return:
    """
    if base_dir is not None:
        set_base_dir(base_dir)

    if type(param_grid) is not list:
        param_grid = [param_grid]

    for params in param_grid:
        if 'task' not in params:
            raise Exception('Please define a task function to execute.')

        if 'num_exp' not in params:
            params['run'] = [0]
        else:
            params['run'] = range(params['num_exp'])
            del params['num_exp']

        if type(params['task']) is not list:
            params['task'] = [params['task']]

        params['gpu_options'] = [tf.GPUOptions(per_process_gpu_memory_fraction=1. / n_processes, allow_growth=True)]

    # Convert parameter grid to iterable list
    params = list(ParameterGrid(param_grid))
    for i in range(len(params)):
        params[i]['id'] = i

    if n_processes > 1:
        Logger.MODE = LogMode.PARALLEL
        with Pool(n_processes) as p:
            p.map(process_task, params)
    else:
        for single_param in params:
            process_task(single_param)

