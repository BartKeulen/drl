import os
import numpy as np


class Variable(metaclass=ABCMeta):

    @abstractmethod
    def sample(self):
        pass


class Constant(Variable):

    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


def print_dict(header, dict_in):
    """
    Prints the keys and values of a dictionary with following markup:

      key: value

    :param header: header to print above the dictonary
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

def print_table(list_of_lists, list_of_headings, indentation='left', color=None):
	if len(list_of_lists) != len(list_of_headings):
		color_print("ERROR: The number of lists and number of headings don't match. Please check your parameters!", color='red', mode='bold')
		exit()

	list_lengths = []
	heading_lengths = []
	lengths = []
	total_length = 0
	for index in range(len(list_of_lists)):
		list_lengths.append(len(max(list_of_lists[index], key=len)))
		heading_lengths.append(len(list_of_headings[index]))

		if(list_lengths[-1] >= heading_lengths[-1]):
			lengths.append(list_lengths[-1])
		else:
			lengths.append(heading_lengths[-1])

		total_length += lengths[-1]

	total_length += 3*(len(list_of_lists) - 1)
	total_length += 4

	color_print("-" * total_length, color=color)
	heading_string = "| "
	for n_heading in range(len(list_of_headings)):
		if(indentation == 'left'):
			heading_string += list_of_headings[n_heading].ljust(lengths[n_heading])
		elif(indentation == 'center'):
			heading_string += list_of_headings[n_heading].center(lengths[n_heading])
		elif(indentation == 'right'):
			heading_string += list_of_headings[n_heading].rjust(lengths[n_heading])
		if(n_heading != len(list_of_headings) - 1):
			heading_string += " | "
	heading_string += " |"

	color_print(heading_string, color=color, mode='bold')
	color_print("-" * total_length, color=color)

	for n_list in range(len(list_of_lists[0])):
		temp_string = "| "
		for cell in range(len(list_of_lists)):
			if(indentation == 'left'):
				temp_string += list_of_lists[cell][n_list].ljust(lengths[cell])
			elif(indentation == 'center'):
				temp_string += list_of_lists[cell][n_list].center(lengths[cell])
			elif(indentation == 'right'):
				temp_string += list_of_lists[cell][n_list].rjust(lengths[cell])
			if(cell != len(list_of_lists) - 1):
				temp_string += " | "
		temp_string += " |"
		color_print(temp_string, color=color)
	color_print("-" * total_length, color=color)