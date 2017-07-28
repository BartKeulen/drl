import json
import os
from glob import glob

import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.utils.extmath import cartesian

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from drl.env.maze import SimpleMaze, Maze, MediumMaze

path = '/home/bartkeulen/results/kdecoverage/'


def load_data(smart_start, env_name, exp_id):
    if smart_start:
        start_type = "smartstart"
    else:
        start_type = "randomstart"
    fp = os.path.join(path, env_name, start_type, "%d" % exp_id)
    files = glob(fp + "/*")

    x = []
    obses = []
    for file in files:
        obs = json.load(open(file, "r"))
        step = int(file.split('/')[-1].split('.')[0])

        x.append(step)
        obses.append(np.array(obs))

    x, obses = zip(*sorted(zip(x, obses), key=lambda x: x[0]))

    return np.array(x), obses


def obs_to_coverage(obses):
    y = []
    for obs in obses:
        img = np.zeros((256, 256))
        obs = (obs + 1.) / 2.
        idxs = (obs * 255).astype('uint')
        for i in range(idxs.shape[0]):
            img[idxs[i, 0], idxs[i, 1]] += 1
        count = np.count_nonzero(img)
        coverage = count / img.size
        y.append(coverage)
    return np.array(y)


def get_runs(smart_start, env_name):
    pass


def create_meshgrid_scores(obs):
    obs = obses[-1]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.15, leaf_size=100)
    kde.fit(obs)

    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    xx, yy = np.meshgrid(x, y)
    xy = cartesian((x, y))
    scores = kde.score_samples(xy)
    scores = np.exp(scores).reshape((x.shape[0], y.shape[0])).transpose()

    return xx, yy, scores

if __name__ == "__main__":
    maze = MediumMaze
    exp_id = 1

    x, obses = load_data(True, maze.__name__, exp_id)
    obs = obses[-1]
    grid_smart = create_meshgrid_scores(obs)

    x, obses = load_data(False, maze.__name__, exp_id)
    obs = obses[-1]
    grid_random = create_meshgrid_scores(obs)

    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(grid_random[0], grid_random[1], grid_random[2],
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.01, 1.01)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Density")

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(grid_smart[0], grid_smart[1], grid_smart[2],
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.01, 1.01)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Density")

    plt.show()
