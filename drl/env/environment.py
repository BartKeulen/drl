from abc import abstractmethod, ABCMeta
import os
import shutil

import gym
from gym.wrappers import Monitor
import pyglet
import ffmpy

from drl.utilities.statistics import get_base_dir, get_summary_dir


MAX_LEN = 10
REC_DIR = '/tmp/drl_record'


class Environment(metaclass=ABCMeta):

    def __init__(self, name):
        self.name = name
        self.observation_space = None
        self.action_space = None
        self.record = False
        self.count = None
        self.episode = None

    @abstractmethod
    def reset(self, x0=None):
        if self.episode is None:
            self.episode = 0
        else:
            self.episode += 1
        self.count = 0

        if self.record:
            save_recording(get_summary_dir(), self.episode-1)
            self.record = False

    def step(self):
        self.count += 1

    @abstractmethod
    def render(self, record=False, close=False):
        pass


class GymEnv(Environment):

    def __init__(self, name, monitor=False):
        super(GymEnv, self).__init__("Gym" + name)
        self.env = gym.make(name)
        if monitor:
            self.gym_dir = os.path.join(get_base_dir(), 'gym')
            self.env = Monitor(self.env, self.gym_dir)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, x0=None):
        super(GymEnv, self).reset()
        return self.env.reset()

    def step(self, action):
        super(GymEnv, self).step()
        return self.env.step(action)

    def render(self, record=False, mode='human', close=False):
        self.record = record
        response = self.env.render(mode=mode, close=close)
        if self.record:
            if not os.path.exists(REC_DIR):
                os.makedirs(REC_DIR)

            path = os.path.join(REC_DIR, '%08d' % self.count)
            pyglet.image.get_buffer_manager().get_color_buffer()
            pyglet.image.get_buffer_manager().get_color_buffer().save(path + '.png')
        return response

    def upload(self, api_key):
        gym.upload(self.gym_dir, api_key)


def save_recording(output_path, filename):
    """
    Saves the recorded frames (.png files) as a mp4 video.

    This can be run from command line using the following command:

        ffmpeg -y -framerate 24 -s 720x720 -i <REC_DIR>/%08d.png <output_path>/<filename>.mp4

    :param output_path:
    :param filename:
    :return:
    """

    # Intialize ffmpy
    ff = ffmpy.FFmpeg(
        inputs={os.path.join(REC_DIR, '%08d.png'): '-y -framerate 24 -s 720x720'},
        outputs={os.path.join(output_path, str(filename) + '.mp4'): None}
    )
    # Send output of ffmpy to log.txt file in temporary record folder
    # if error check the log file
    ff.run(stdout=open(os.path.join(REC_DIR, 'log.txt'), 'w'), stderr=open(os.path.join(REC_DIR, 'tmp.txt'), 'w'))
    # Remove .png and log.txt file
    shutil.rmtree(REC_DIR)