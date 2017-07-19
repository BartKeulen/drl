from multiprocessing import Pool
import tensorflow as tf
from sklearn.model_selection import ParameterGrid


class Mode:
    LOCAL = 'local'
    REMOTE = 'remote'


def process_task(params):
    if params['parallel']:
        print('Task %d started with settings: %s' % (params['id'], params))
    graph = tf.Graph()

    with graph.as_default():
        sess = tf.Session(config=tf.ConfigProto(gpu_options=params['gpu_options']))

        agent = params['task'](params)

        if agent.__class__.__name__ != 'RLAgent':
            raise Exception('The defined task function has to return a RLAgent object.')

        agent.train(sess, params['run'], parallel=params['parallel'], mode=params['mode'])

        sess.close()

        if params['parallel']:
            print('Task %d finished' % params['id'])


def run_experiment(param_grid, n_processes=1, mode=Mode.LOCAL):
    """
    Method used for running a reinforcement learning experiment.

    :param param_grid: parameter grid defining tasks and parameter, see explanation above.
    :param n_processes: when running in parallel, the number of parallel processes is defined here
    :return:
    """
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

        if n_processes > 1:
            params['parallel'] = [True]

    # Convert parameter grid to iterable list
    params = list(ParameterGrid(param_grid))
    for i in range(len(params)):
        params[i]['id'] = i

    if n_processes > 1:
        with Pool(n_processes) as p:
            p.map(process_task, params)
    else:
        for single_param in params:
            process_task(single_param)

