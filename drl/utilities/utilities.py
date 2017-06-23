import os
import numpy as np


def print_dict(header, dict_in):
    """
    Prints the keys and values of a dictionary with following markup:

      key: value

    :param dict_in: dictonary to print
    """
    len_key = max([len(key) for key in dict_in.keys()]) + 5
    len_value = max([len(str(value)) for value in dict_in.values()]) + 5

    print('')
    print('-'*(len_key+len_value))
    print('\033[1m{:s}\033[0m'.format(header))
    print('='*(len_key+len_value))
    for key in dict_in.keys():
        print('{:{width1}s}{:{width2}s}'.format(key, str(dict_in[key]), width1=len_key, width2=len_value))
    print('-'*(len_key+len_value))


def func_serializer(x, u, func, first=True, second=True):
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
    fx = []
    fu = []
    fxx = []
    fuu = []
    fxu = []
    for i in range(x.shape[0]):
        res = func(x[i], u[i])
        out.append(res[0])
        if first:
            fx.append(res[1])
            fu.append(res[2])
        if second:
            fxx.append(res[3])
            fuu.append(res[4])
            fxu.append(res[5])

    if not first:
        fx = np.NaN
        fu = np.NaN
    if not second:
        fxx = np.NaN
        fuu = np.NaN
        fxu = np.NaN

    return np.array(out), np.array(fx), np.array(fu), np.array(fxx), np.array(fuu), np.array(fxu)

