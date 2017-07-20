import time
import os
import glob
import inspect
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import drl
from .utilities import print_dict

DIR = os.path.join(os.path.dirname(inspect.getfile(drl)), '../results')


class Statistics(object):

    def __init__(self,
                 env,
                 algo,
                 rl_agent,
                 base_dir=None,
                 save=False,
                 print=True):
        self.env = env
        self.algo = algo
        self.info, self.algo_options = algo.get_info()
        try:
            self.info['env'] = self.env.env.spec.id
        except:
            self.info['env'] = self.env.__class__.__name__
        self.agent_info = rl_agent.get_info()
        self.save = save
        self.print = print

        # Create summary directory
        timestamp = time.strftime('%Y%m%d%H%M')
        if base_dir is None:
            base_dir = DIR
        self.summary_dir = get_summary_dir(base_dir, self.info['env'], self.info['algo'], timestamp, save)
        self.info['timestamp'] = timestamp

        # Initialize tags
        self.tags = self.algo.TAGS + ['reward']

    def reset(self, run=None):
        # Directory to save results
        if run is not None:
            self.summary_dir = os.path.join(self.summary_dir, '%d' % run)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.info['run'] = run

        # Save info and options
        info = {
            'info': self.info,
            'agent': self.agent_info,
            'algo': self.algo_options
        }
        pickle.dump(info, open(os.path.join(self.summary_dir, 'info.p'), 'wb'))

        # Summary variables
        self.summary = {
            'episodes': [],
            'steps': [],
            'time': [],
            'values': {}
        }
        for tag in self.tags:
            self.summary['values'][tag] = []

        # Get starting time
        self.start = time.time()

        return self.summary_dir

    def save_episode(self, episode, steps, update_info):
        self.summary['episode'].append(episode)
        self.summary['steps'].append(steps)
        self.summary['time'].append(time.time() - self.start)

        # Update summary variables
        for tag, value in update_info:
            self.summary['values'][tag].append(np.mean(value))
        pickle.dump(self.summary, open(os.path.join(self.summary_dir, 'summary.p'), 'wb'))


def get_summary_dir(dir_name, env_name, algo_name, timestamp, save=False):
    """
    Function for generating directory for storing summary results of tensorflow session.
    If directory does not exist one is created.

    :param dir_name: Base directory
    :param env_name:
    :param algo_name:
    :param timestamp:
    :param save: Boolean determining if values should be stored in temporary folder or not
                - True, keep files
                - False, put them in temporary folder
    :return: Directory for storing summary results
    """

    if save:
        tmp = 'eval'
    else:
        tmp = 'tmp'

    summary_dir = os.path.join(dir_name, tmp, env_name, algo_name, timestamp)

    paths = glob.glob(summary_dir + "*")
    if len(paths) == 0:
        os.makedirs(summary_dir)
    else:
        count = 1
        for path in paths:
            if os.path.isdir(path):
                dir_name = path.split('/')[-1]
                if dir_name != timestamp:
                    idx = int(path.split('_')[-1])
                    if idx >= count:
                        count = idx + 1
        summary_dir = "%s_%d" % (summary_dir, count)
        os.makedirs(summary_dir)

    return summary_dir