from multiprocessing import Pool
from sklearn.model_selection import ParameterGrid
import tensorflow as tf


class Parallel(object):

    def __init__(self, task, param_grid=None, n_processes=4):
        if param_grid is not None:
            params = list(ParameterGrid(param_grid))
        else:
            params = [{'run': 0}]

        for i in range(len(params)):
            params[i]['process_id'] = i

        self.task = task
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1. / n_processes, allow_growth=True)

        with Pool(n_processes) as p:
            p.map(self.process_task, params)

    def process_task(self, params):
        print('Process %d started with settings: %s' % (params['process_id'], params))
        graph = tf.Graph()

        with graph.as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))

            agent = self.task(params)

            agent.train(sess, params['run'], parallel=True)

            sess.close()

            print('Process %d finished' % params['process_id'])
