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

	print('-' * (len_key + len_value))
	print('\033[1m{:s}\033[0m'.format(header))
	print('=' * (len_key + len_value))
	for key in dict_in.keys():
		print('{:{width1}s}{:{width2}s}'.format(key, str(dict_in[key]), width1=len_key, width2=len_value))
	print('-' * (len_key + len_value))
	print('')


def get_summary_dir(dir_name, env_name, algo_name, save=False):
	"""
    Function for generating directory for storing summary results of tensorflow session.
    If directory does not exist one is created.

    :param dir_name: Base directory
    :param env_name:
    :param algo_name:
    :param save: Boolean determining if values should be stored in temporary folder or not
                - True, keep files
                - False, put them in temporary folder
    :return: Directory for storing summary results
    """

	if save:
		tmp = 'eval'
	else:
		tmp = 'tmp'

	summary_dir = os.path.join(dir_name, tmp, env_name, algo_name)

	count = 0
	for f in os.listdir(summary_dir):
		child = os.path.join(summary_dir, f)
		if os.path.isdir(child):
			count += 1

	summary_dir = os.path.join(summary_dir, str(count))

	if not os.path.exists(summary_dir):
		os.makedirs(summary_dir)

	return summary_dir


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


def color_print(text, color=None, mode=None):
	colors = {
		'pink': '\033[95m',
		'blue': '\033[94m',
		'green': '\033[92m',
		'yellow': '\033[93m',
		'red': '\033[91m',
	}

	modes = {
		'bold': '\033[1m',
		'underline': '\033[4m',
		'bold_underline': '\033[1m\033[4m'
	}

	reset = '\033[0m'

	if mode is None and color in colors:
		print(colors[color] + text + reset)
	elif color is None and mode in modes:
		print(modes[mode] + text + reset)
	elif mode in modes and color in colors:
		print(modes[mode] + colors[color] + text + reset)
	else:
		print(modes['bold'] + colors['red'] + 'Incorrect inputs to color_print. Displaying your text normally' + reset)
		print(text)
