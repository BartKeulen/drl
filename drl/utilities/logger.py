from tqdm import tqdm


class LogMode(object):
    PBAR = 0
    PARALLEL = 1


class Logger(object):
    MODE = LogMode.PBAR

    def __init__(self, *args, **kwargs):
        if Logger.MODE == LogMode.PBAR:
            self.logger = PbarLogger(*args, **kwargs)
        elif Logger.MODE == LogMode.PARALLEL:
            self.logger = ParallelLogger(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.logger.update(*args, **kwargs)

    def write(self, message, color=None, mode=None):
        self.logger.write(message, color, mode)


class ParallelLogger(object):

    def __init__(self, total, prefix=None):
        self.prefix = prefix
        self.count = 0

    def update(self, n, *args, **kwargs):
        # TODO: Fix logger for parallel execution
        if self.prefix is not None:
            print_str = self.prefix + " - Episode: %d, " % self.count
        else:
            print_str = "Episode: %d" % self.count
        for key, value in kwargs.items():
            print_str += "{:s}: {:.2f}, ".format(key, value)
        print(print_str)
        self.count += n

    def write(self, message, color=None, mode=None):
        print(decorate_message(message, color, mode))


class PbarLogger(object):

    def __init__(self, total, prefix=None):
        self.pbar = tqdm(total=total, desc=prefix)

    def update(self, n=1, **kwargs):
        if n == -1:
            self.pbar.close()
        else:
            self.pbar.set_postfix(kwargs)
            self.pbar.update(n)

    def write(self, message, color=None, mode=None):
        tqdm.write(message)
        # self.pbar.write(decorate_message(message, color, mode))


def decorate_message(message, color=None, mode=None):
    if color is None and mode is None:
        return message

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
        return colors[color] + message + reset
    elif color is None and mode in modes:
        return modes[mode] + message + reset
    elif mode in modes and color in colors:
        return modes[mode] + colors[color] + message + reset
    else:
        return modes['bold'] + colors['red'] + 'Incorrect inputs to color_print. Displaying your text normally' + \
               reset + '\n' + message
