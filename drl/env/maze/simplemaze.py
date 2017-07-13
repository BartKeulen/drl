from Box2D import *
import numpy as np
from gym import spaces

import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_RIGHT, K_LEFT


class MediumMaze(object):
    FORCE_SCALE = 100.
    PPM = 20.0
    TARGET_FPS = 60
    TIME_STEP = 1.0 / TARGET_FPS
    SCREEN_WIDTH, SCREEN_HEIGHT = int(50*PPM), int(32*PPM)

    def __init__(self):
        self._world = b2World(gravity=(0, 0))

        self._walls = []
        positions = [(25, 2), (25, 30), (2, 15), (48, 15), (18, 15)]
        sizes = [()]