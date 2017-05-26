import tensorflow as tf
import logging
from utils import get_summary_dir
import time

DIR = '/home/bartkeulen/results/'


class Statistics(object):

    def __init__(self,
                 sess,
                 env_name,
                 algo,
                 summary_tags,
                 settings=None,
                 save=False,
                 update_repeat=1):
        self.sess = sess
        self.env_name = env_name
        self.update_repeat = update_repeat
        self.summary_tags = summary_tags
        self.count = 0
        self.start_time = None

        # Init logger
        self.logger = logging.getLogger(algo)
        self.logger.setLevel(logging.DEBUG)

        # Init directory and writer
        self.summary_dir = get_summary_dir(DIR, env_name, algo, settings, save)
        self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        self.logger.info("For visualizing run:\n  tensorboard --logdir=%s" % self.summary_dir)

        # Init variables
        with tf.variable_scope('summary'):
            self.summary_values = {}
            self.summary_placeholders = {}
            self.summary_ops = {}

            self.summary_placeholders['reward'] = tf.placeholder('float32', None, name='reward')
            self.summary_ops['reward'] = tf.summary.scalar('%s/reward' % self.env_name, self.summary_placeholders['reward'])
            self.summary_placeholders['ave_r'] = tf.placeholder('float32', None, name='ave_r')
            self.summary_ops['ave_r'] = tf.summary.scalar('%s/average_reward' % self.env_name, self.summary_placeholders['ave_r'])
            for tag in summary_tags:
                self.summary_values[tag] = 0.
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar('%s/%s' % (self.env_name, tag), self.summary_placeholders[tag])

    def get_tags(self):
        return self.summary_tags

    def reset(self):
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
        log_str = '| episode: %d | steps: %d | reward: %.2f | ave r: %.3f |' % (episode, step, reward, reward / step)

        if self.count == 0:
            self.count = 1

        for tag, value in self.summary_values.items():
            log_str += ' %s: %.2f |' % (tag, value / self.count)

        log_str += ' time elapsed: %.f sec' % (time.time() - self.start_time)
        self.logger.info(log_str)

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
