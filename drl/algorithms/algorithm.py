from abc import ABCMeta, abstractmethod
import os

import tensorflow as tf

from drl.utilities.statistics import get_summary_dir


class Algorithm(metaclass=ABCMeta):

    @abstractmethod
    def train(self, sess):
        pass

    def save_model(self, checkpoint=None):
        """
        Saves the current Tensorflow variables in the specified path, after saving the location is printed.
        All Tensorflow variables are saved, this means you can even continue training if you want.

        :param path: location to save the model
        """
        # TODO: ADD saving the full information of the experiment
        path = os.path.join(get_summary_dir(), 'model')
        saver = tf.train.Saver()
        saver.save(self.sess, path, global_step=checkpoint)

    def restore(self, path, checkpoint=None):
        """
        Restores the Tensorflow variables saved at the specified path.
        :param path: location of the saved model
        """
        # TODO: Add the rest of restore method so an experiment can be fully restored with all settings from file
        saver = tf.train.Saver()
        path = os.path.join(path, 'model')
        if checkpoint is None:
            saver.restore(self.sess, path)
        else:
            saver.restore(self.sess, path + "-" + checkpoint)