import time
import os
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
                 save=False):
        self.env = env
        self.algo = algo
        self.info, self.algo_options = algo.get_info()
        self.info['env'] = self.env.env.spec.id
        self.options = rl_agent.get_info()
        self.save = save

        # Init directory
        if base_dir is None:
            dir = DIR
        else:
            dir = base_dir

        if save:
            tmp = 'eval'
        else:
            tmp = 'tmp'

        timestamp = time.strftime('%Y/%m/%d/%H%M')

        self.base_dir = os.path.join(dir, tmp, self.info['env'], self.info['name'], timestamp)

        self.info['timestamp'] = timestamp

    def reset(self, run=0):
        # Directory to save results
        self.summary_dir = os.path.join(self.base_dir, 'run_%d' % run)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.info['run'] = run

        # Save info and options
        info = {
            'info': self.info,
            'agent': self.options,
            'algo': self.algo_options
        }
        pickle.dump(info, open(os.path.join(self.summary_dir, 'info.p'), 'wb'))

        # Summary variables
        self.summary = {
            'episode': [],
            'values': {
                'reward': [],
                'average reward': []
            },
        }
        self.summary_tmp = {}
        for tag in self.info['summary_tags']:
            self.summary['values'][tag] = []
            self.summary_tmp[tag] = []

        # Print info
        self.print_info()

        # tqdm progress bar for total process
        self.pbar_tot = tqdm(total=self.options['num_episodes'], desc='{:>7s}'.format('Total'))

        return self.summary_dir

    def print_info(self):
        print_dict('Agent options', self.options,)
        print_dict('Algorithm options', self.algo_options)
        self.algo.print_summary()

        ndash = '-' * 50
        print('\n\033[1m{:s} Start training {:s}\033[0m\n'.format(ndash, ndash))
        print('Summary directory: \n   {:s}\n'.format(self.summary_dir))

    def update(self, reward, update_info):
        # Update episode progress bar
        self.pbar_ep.set_postfix(reward='{:.2f}'.format(reward), **update_info)
        self.pbar_ep.update()

        # Update summary variables
        for tag, value in update_info.items():
            self.summary_tmp[tag].append(value)

    def ep_reset(self):
        # Reset episode progress bar
        self.pbar_ep = tqdm(total=self.options['max_steps'], desc='{:>7s}'.format('Episode'), leave=False)

        # Reset summary variables
        for tag in self.summary_tmp:
            self.summary_tmp[tag] = []

    def write(self, episode, steps, reward):
        # Update total progress bar and close episode progress bar
        self.pbar_ep.close()
        self.pbar_tot.set_postfix(reward='{:.2f}'.format(reward), steps='{:d}'.format(steps))
        self.pbar_tot.update()

        # Update summary variables
        self.summary['episode'].append(episode)
        self.summary['values']['reward'].append(reward)
        self.summary['values']['average reward'].append(reward / steps)

        for tag, value in self.summary_tmp.items():
            self.summary['values'][tag].append(np.mean(value))

        pickle.dump(self.summary, open(os.path.join(self.summary_dir, 'summary.p'), 'wb'))


class StatisticsTF(object):
    """
    Statistics is used for saving data from a tensorflow session.
    After each episodes the results are saved in summary directory and the episode results are printed on stdout.
    """

    def __init__(self,
                 sess,
                 env_name,
                 algo_info,
                 summary_dir=None,
                 save=False,
                 update_repeat=1):
        self.sess = sess
        self.env_name = env_name
        self.algo_name = algo_info['name']
        self.summary_tags = algo_info['summary_tags']
        self.save = save
        self.update_repeat = update_repeat
        self.count = 0
        self.start_time = None

        # Init directory
        if summary_dir is None:
            dir = DIR
        else:
            dir = summary_dir
        self.summary_dir = get_summary_dir(dir, self.env_name, self.algo_name, self.save)

        print("For visualizing run:\n  tensorboard --logdir=%s\n" % os.path.abspath(self.summary_dir))

        # Init variables
        with tf.variable_scope(self.env_name):
            self.summary_values = {}
            self.summary_placeholders = {}
            self.summary_ops = {}

            self.summary_placeholders['reward'] = tf.placeholder('float32', None, name='reward')
            self.summary_ops['reward'] = tf.summary.scalar('reward', self.summary_placeholders['reward'])
            self.summary_placeholders['ave_r'] = tf.placeholder('float32', None, name='ave_r')
            self.summary_ops['ave_r'] = tf.summary.scalar('average_reward', self.summary_placeholders['ave_r'])
            for tag in self.summary_tags:
                self.summary_values[tag] = 0.
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar('%s' % tag, self.summary_placeholders[tag])

        # # Write options to options.txt file
        # json_string = json.dumps(algo_info[1])
        # f = open(os.path.join(self.summary_dir, 'options.txt'), 'w')
        # f.write(json_string)
        # f.close()

    def reset(self, run):
        run_dir = os.path.join(self.summary_dir, 'run_%d' % run)
        self.writer = tf.summary.FileWriter(run_dir)
        return run_dir

    def get_tags(self):
        return self.summary_tags

    def episode_reset(self):
        if self.start_time is None:
            self.start_time = time.time()

        for tag in self.summary_tags:
            self.summary_values[tag] = 0.
        self.count = 0

    def update(self, summary_updates):
        if len(list(set(summary_updates.keys()) & set(self.summary_tags))) != len(self.summary_tags):
            raise Exception("tags in update are different from tags summary tags.")

        for tag in self.summary_tags:
            self.summary_values[tag] += summary_updates[tag]

        self.count += 1

    def write(self, reward, episode, step):
        log_str = '[TRAIN] episode: {:>6d} | steps: {:>6d} | reward: {:>10.2f} | ave r: {:>6.2f} |'.format(episode, step, reward, reward / step)

        if self.count == 0:
            self.count = 1

        for tag, value in self.summary_values.items():
            log_str += ' {:s}: {:>6.2f} |'.format(tag, value / self.count)

        log_str += ' time elapsed: {:>6.0f} sec'.format(time.time() - self.start_time)
        print(log_str)

        summary_str_list = self.sess.run([self.summary_ops[tag] for tag in self.summary_values.keys()], {
            self.summary_placeholders[tag]: value / self.count for tag, value in
            self.summary_values.items()
        })

        summary_str_list.append(self.sess.run(self.summary_ops['reward'], {
            self.summary_placeholders['reward']: reward
        }))
        summary_str_list.append(self.sess.run(self.summary_ops['ave_r'], {
            self.summary_placeholders['ave_r']: reward / step
        }))

        for summary_str in summary_str_list:
            self.writer.add_summary(summary_str, episode)

        self.writer.flush()


def get_summary_dir(dir_name, env_name, algo_name, save=False):
    """
    Function for generating directory for storing summary results of tensorflow session.
    If directory does not exist one is created.

    :param dir_name: Base directory
    :param env_name:
    :param algo_name:
    :param save: Boolean determining if values should be stored in temporary folder or not
                - True, keep files
                - False, put them in temporary folder
    :return: Directory for storing summary results
    """

    if save:
        tmp = 'eval'
    else:
        tmp = 'tmp'

    timestamp = time.strftime('%Y/%m/%d/%H%M')

    summary_dir = os.path.join(dir_name, tmp, env_name, algo_name, timestamp)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # count = 0
    # for f in os.listdir(summary_dir):
    #     child = os.path.join(summary_dir, f)
    #     if os.path.isdir(child):
    #         count += 1
    #
    # summary_dir = os.path.join(summary_dir, str(count))
    #
    # if not os.path.exists(summary_dir):
    #     os.makedirs(summary_dir)

    return summary_dir