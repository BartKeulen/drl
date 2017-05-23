import tensorflow as tf
import logging
from deepreinforcementlearning.utils import get_summary_dir

DIR = '/home/bartkeulen/results/'


class Statistics(object):

    def __init__(self,
                 sess,
                 env_name,
                 algo,
                 summary_tags,
                 update_repeat=1):
        self.sess = sess
        self.env_name = env_name
        self.update_repeat = update_repeat
        self.summary_tags = summary_tags

        # Init logger
        self.logger = logging.getLogger(algo)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s', datefmt='%I:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Init directory and writer
        self.summary_dir = get_summary_dir(DIR, env_name, algo)
        self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        self.logger.info("For visualizing run:\n  tensorboard --logdir=%s" % self.summary_dir)

        # Init variables
        with tf.variable_scope('summary'):
            self.summary_values = {}
            self.summary_placeholders = {}
            self.summary_ops = {}

            self.summary_placeholders['reward'] = tf.placeholder('float32', None, name='Reward')
            self.summary_ops['reward'] = tf.summary.scalar('%s/reward' % self.env_name, self.summary_placeholders['reward'])
            for tag in summary_tags:
                self.summary_values[tag] = 0.
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar('%s/%s' % (self.env_name, tag), self.summary_placeholders[tag])

    def get_tags(self):
        return self.summary_tags

    def reset(self):
        for tag in self.summary_tags:
            self.summary_values[tag] = 0.

    def update(self, summary_updates):
        if len(list(set(summary_updates.keys()) & set(self.summary_tags))) != len(self.summary_tags):
            raise Exception("tags in update are different from tags summary tags.")


        for tag in self.summary_tags:
            self.summary_values[tag] += summary_updates[tag]

    def write(self, reward, episode, step):
        log_str = '| episode: %d | steps: %d | reward: %.3f | ave r: %.3f |' % (episode, step, reward, reward / step)
        for tag, value in self.summary_values.items():
            log_str += ' %s: %.3f |' % (tag, value / step / self.update_repeat)
        self.logger.info(log_str)

        summary_str_list = self.sess.run([self.summary_ops[tag] for tag in self.summary_values.keys()], {
            self.summary_placeholders[tag]: value / step / self.update_repeat for tag, value in
            self.summary_values.items()
        })

        summary_str_list.append(self.sess.run(self.summary_ops['reward'], {
            self.summary_placeholders['reward']: reward
        }))

        for summary_str in summary_str_list:
            self.writer.add_summary(summary_str, episode)

        self.writer.flush()
