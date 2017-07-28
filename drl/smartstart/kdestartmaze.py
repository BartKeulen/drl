import pygame
from pygame.locals import *
import numpy as np


from drl.env.maze import Maze
from drl.replaybuffer import ReplayBufferKD

render = True
max_steps = 5000


if __name__ == "__main__":
    x = input("Use smart start? [y/n] ")

    if x == "y":
        smart_start = True
    else:
        smart_start = False

    x = input("Which maze to use? [simple, medium] ")
    if x == "simple":
        env = Maze.generate_maze(Maze.SIMPLE)
        max_steps = 5000
    elif x == "medium":
        env = Maze.generate_maze(Maze.MEDIUM)
        max_steps = 25000
    else:
        raise Exception("Please choose from the available environments: [simple, medium]")

    print("Using smart start: %s, and %s environment" % (smart_start, env.__class__.__name__))

    buffer = ReplayBufferKD(100000)

    obs_t = env.reset()
    if render:
        env.render()

    running = True
    steps = 0

    finishes = []
    while running:
        action = np.random.randn(2)

        if render:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

        obs_tp1, r, t, _ = env.step(action)
        steps += 1

        buffer.add(obs_t, action, r, obs_tp1, t)

        obs_t = obs_tp1

        if t or steps >= max_steps:
            print("Finished in %d steps" % steps)
            finishes.append(steps)
            steps = 0

            if smart_start:
                samples, scores = buffer.kd_estimate(100)
                argmin_score = np.argmin(scores)
                x0 = samples[argmin_score]
            else:
                x0 = buffer.sample(1)[0][0]
                samples = None
            obs_t = env.reset(x0)
            env.render(close=(not running), data=buffer.get_rgb_array(env), samples=samples)

        if len(finishes) == 25:
            running = False

    print("Smart start: %s, average steps: %s " % (smart_start, np.average(finishes)))

    input("Press enter to exit")