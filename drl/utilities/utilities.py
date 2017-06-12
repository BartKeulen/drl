import os
import numpy as np


def get_summary_dir(dir_name, env_name, algo_name, settings=None, save=False):
    """
    Function for generating directory for storing summary results of tensorflow session.
    If directory does not exist one is created.

    :param dir_name: Base directory
    :param env_name:
    :param algo_name:
    :param settings: If given settings are added as subfolders
    :param save: Boolean determining if values should be stored in temporary folder or not
                - True, keep files
                - False, put them in temporary folder
    :return: Directory for storing summary results
    """

    if save:
        tmp = 'eval'
    else:
        tmp = 'test'

    summary_dir = os.path.join(dir_name, tmp, env_name, algo_name)

    if settings is None:
        summary_dir = os.path.join(summary_dir, 'other')
    else:
        for key, value in settings.items():
            summary_dir = os.path.join(summary_dir, '%s=%s' % (key, value))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    count = 0
    for f in os.listdir(summary_dir):
        child = os.path.join(summary_dir, f)
        if os.path.isdir(child):
            count += 1

    return os.path.join(summary_dir, str(count))


def func_serializer(x, u, func):
    """
    Helper function for serializing a function.
    Calculates the value of the function for each (state, control input) pair.

    Function must be of the form:
        y = f(x, u)


    :param x: vector containing states
    :param u: vector containing control inputs
    :param func:
    :return: numpy array containing outputs
    """
    out = []
    for i in range(x.shape[0]):
        out.append(func(x[i], u[i]))
    return np.array(out)