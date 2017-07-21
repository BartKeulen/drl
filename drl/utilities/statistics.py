import time
import datetime
import os
import glob
import inspect
import pickle
from collections import Counter
import numpy as np

import drl

BASE_DIR = os.path.join(os.path.dirname(inspect.getfile(drl)), '../results')


class Statistics(object):

    def __init__(self,
                 env_name,
                 algo_name,
                 run=None,
                 tags=None,
                 save=False,
                 base_dir=None):
        timestamp = datetime.datetime.now()

        # Create summary directory
        if base_dir is None:
            base_dir = BASE_DIR
        self.summary_dir = get_summary_dir(base_dir, env_name, algo_name, save)

        if run is not None:
            self.summary_dir = os.path.join(self.summary_dir, '%d' % run)

        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        # tags and summary variables
        self.tags = []
        self.summary = {
            'env': env_name,
            'algo': algo_name,
            'timestamp': timestamp,
            'run': run,
            'episodes': [],
            'steps': [],
            'time': [],
            'values': []
        }

        self.add_tags(tags)

        # Get starting time
        self.start = time.time()

    def add_tags(self, tags):
        if tags is None:
            return

        if type(tags) is not list:
            self.tags.append(tags)
        else:
            self.tags += tags

        for tag in self.tags:
            self.summary['values'].append({'x': [], 'y': [], 'name': tag})

    def update_tags(self, episode, tags, values):
        for tag, value in zip(tags, values):
            self.update_tag(episode, tag, value)

    def update_tag(self, episode, tag, value):
        if tag not in self.tags:
            raise Exception("Tag does not correspond to defined tags, please add tag using the add_tags function")

        for summary in self.summary['values']:
            if tag == summary['name']:
                summary['x'].append(episode)
                summary['y'].append(value)

    def save_episode(self, episode, steps):
        self.summary['episodes'].append(episode)
        self.summary['steps'].append(steps)
        self.summary['time'].append(time.time() - self.start)

        pickle.dump(self.summary, open(os.path.join(self.summary_dir, 'summary.p'), 'wb'))


def get_summary_dir(dir_name, env_name, algo_name, save=False):
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
    if dir_name is None:
        dir_name = BASE_DIR

    if save:
        tmp = 'eval'
    else:
        tmp = 'tmp'

    summary_dir = os.path.join(dir_name, tmp, env_name, algo_name)

    paths = glob.glob(summary_dir + "_*")
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