from Box2D import *
import numpy as np
from gym import spaces

import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_RIGHT, K_LEFT

from drl.env.maze.maze import Maze


class MediumMaze(Maze):

    def __init__(self):
        world_size = (50, 50)
        wall_pos = [(25, 2), (25, 48), (2, 25), (48, 25), (18, 18), (32, 32)]
        wall_sizes = [(25, 2), (25, 2), (2, 25), (2, 25), (18, 2), (18, 2)]
        init_pos = (9, 9)
        goal = (41, 41)

        super(MediumMaze, self).__init__(world_size, wall_pos, wall_sizes, init_pos, goal)

if __name__ == "__main__":
    env = MediumMaze()
    env.reset()
    env.render()

    running = True
    steps = 0
    while running:
        action = [0, 0]
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

            if event.type == KEYDOWN:
                if event.key == K_UP:
                    action = [0, 1]
                if event.key == K_DOWN:
                    action = [0, -1]
                if event.key == K_LEFT:
                    action = [-1, 0]
                if event.key == K_RIGHT:
                    action = [1, 0]

        obs, r, t, _ = env.step(action)
        steps += 1

        if t:
            obs = env.reset()
            print("Number of steps: %d" % steps)
            steps = 0

        print("Obs: %s, r: %.2f, t: %d" % (obs, r, t))

        env.render(close=(not running))
