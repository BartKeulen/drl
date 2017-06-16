import tensorflow as tf
from .utilities import get_summary_dir, print_dict
import time
import os
import inspect
import drl
import json

DIR = os.path.join(os.path.dirname(inspect.getfile(drl)), '../results')


class Statistics(object):
    """
    Statistics is used for saving data from a tensorflow session.
    After each episodes the results are saved in summary directory and the episode results are printed on stdout.
    """

    def __init__(self,
                 sess,
                 env_name,
                 algo,
                 summary_dir=None,
                 save=False,
                 update_repeat=1):
        self.sess = sess
        self.env_name = env_name
        self.algo_name = algo[0]['name']
        self.summary_tags = algo[0]['summary_tags']
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

        # Write options to options.txt file
        json_string = json.dumps(algo[1])
        f = open(os.path.join(self.summary_dir, 'options.txt'), 'w')
        f.write(json_string)
        f.close()

        print("Training options:")
        print_dict(algo[1])

    def reset(self, run):
        run_dir = os.path.join(self.summary_dir, 'run_%d' % run)
        self.writer = tf.summary.FileWriter(run_dir)

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
        log_str = 'episode: {:>6d} | steps: {:>6d} | reward: {:>10.2f} | ave r: {:>6.2f} |'.format(episode, step, reward, reward / step)

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
