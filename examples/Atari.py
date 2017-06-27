"""
Install opencv with ffmpeg and gtk2 support to use this script.
In order to maintain compatibility, conda-forge opencv builds without gtk2 support.

Use the following command with anaconda:
conda install --channel loopbio --channel conda-forge --channel pkgw-forge gtk2 ffmpeg ffmpeg-feature gtk2-feature opencv openblas
"""

from ale_python_interface import ALEInterface
import numpy as np
import cv2
from random import randrange

class Atari:
    def __init__(self,rom_name):
        self.ale = ALEInterface()
        self.max_frames_per_episode = self.ale.getInt(b"max_num_frames_per_episode")
        self.ale.setInt(b"random_seed",123)
        self.ale.setInt(b"frame_skip",4)
        self.ale.loadROM(rom_name)
        self.screen_width,self.screen_height = self.ale.getScreenDims()
        self.legal_actions = self.ale.getMinimalActionSet()
        self.action_map = dict()
        for i in range(len(self.legal_actions)):
            self.action_map[self.legal_actions[i]] = i
        print(len(self.legal_actions))
        self.windowname = rom_name.decode() # Convert bytes to string
        cv2.startWindowThread()
        cv2.namedWindow(self.windowname)

    def preprocess(self, image):
        image = cv2.cvtColor(cv2.resize(image, (84, 110)), cv2.COLOR_BGR2GRAY)
        image = image[26:110,:]
        ret, image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
        return np.reshape(image,(84,84, 1))

    def get_image(self):
        numpy_surface = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
        self.ale.getScreenRGB(numpy_surface)
        image = np.reshape(numpy_surface, (self.screen_height, self.screen_width, 3))
        return self.preprocess(image)

    def newGame(self):
        self.ale.reset_game()
        return self.get_image()

    def next(self, action):
        reward = self.ale.act(self.legal_actions[np.argmax(action)])
        nextstate = self.get_image()

        cv2.imshow(self.windowname, nextstate)
        if self.ale.game_over():
            self.newGame()
        #print "reward %d" % reward
        return nextstate, reward, self.ale.game_over()