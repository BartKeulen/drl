import time
import datetime
import os
import inspect
import json
from glob import glob

import drl

BASE_DIR = os.path.join(os.path.dirname(inspect.getfile(drl)), '../results/', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
SUMMARY_DIR = None


def set_base_dir(path):
    global BASE_DIR
    BASE_DIR = path


def get_base_dir():
    return BASE_DIR


def set_summary_dir(env_name, algo_name):
    global SUMMARY_DIR

    summary_dir = os.path.join(BASE_DIR, env_name, algo_name)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    count = 0
    for sub_dir in glob(summary_dir + '/*'):
        if os.path.isdir(sub_dir):
            dir_idx = int(sub_dir.split('/')[-1])
            if dir_idx >= count:
                count = dir_idx + 1
    summary_dir = os.path.join(summary_dir, str(count))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    SUMMARY_DIR = summary_dir
    return summary_dir


def get_summary_dir():
    return SUMMARY_DIR


def save(file_name, value):
    path = os.path.join(get_summary_dir(), file_name + '.json')
    with open(path, 'w') as fp:
        json.dump(value, fp)


class Statistics(object):

    def __init__(self,
                 env_name,
                 algo_name,
                 tags=None,
                 base_dir=None):
        # Create summary directory
        if base_dir is not None:
            set_base_dir(base_dir)

        # Set the global summary directory
        set_summary_dir(env_name, algo_name)

        # tags and summary variables
        self.tags = []
        self.summary = {
            'env': env_name,
            'algo': algo_name,
            'timestamp': datetime.datetime.now().isoformat(),
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

    def write(self):
        save('summary', self.summary)
