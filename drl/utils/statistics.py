import tensorflow as tf
from .utils import get_summary_dir
import time
import os

DIR = './results/'


class Statistics(object):
    """
    Statistics is used for saving data from a tensorflow session.
    After each episodes the results are saved in summary directory and the episode results are printed on stdout.
    """

    def __init__(self,
                 sess,
                 env_name,
                 algo,
                 summary_tags,
                 res_dir=None,
                 settings=None,
                 save=False,
                 update_repeat=1):
        self.sess = sess
        self.env_name = env_name
        self.update_repeat = update_repeat
        self.summary_tags = summary_tags
        self.count = 0
        self.start_time = None

        # Init directory and writer
        if res_dir is not None:
            dir = res_dir
        else:
            dir = DIR
        self.summary_dir = get_summary_dir(dir, env_name, algo, settings, save)
        self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        print("For visualizing run:\n  tensorboard --logdir=%s\n" % os.path.abspath(self.summary_dir))

        # Init variables
        with tf.variable_scope(env_name):
            self.summary_values = {}
            self.summary_placeholders = {}
            self.summary_ops = {}

            self.summary_placeholders['reward'] = tf.placeholder('float32', None, name='reward')
            self.summary_ops['reward'] = tf.summary.scalar('reward', self.summary_placeholders['reward'])
            self.summary_placeholders['ave_r'] = tf.placeholder('float32', None, name='ave_r')
            self.summary_ops['ave_r'] = tf.summary.scalar('average_reward', self.summary_placeholders['ave_r'])
            for tag in summary_tags:
                self.summary_values[tag] = 0.
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar('%s' % tag, self.summary_placeholders[tag])

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

        log_str += ' time elapsed: %.f sec |' % (time.time() - self.start_time)
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
