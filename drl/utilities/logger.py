from tqdm import tqdm


class LogMode(object):
    PBAR = 0
    INFO = 1
    SHORT = 2


class Logger(object):
    MODE = LogMode.PBAR

    def __init__(self, *args, **kwargs):
        if Logger.MODE == LogMode.PBAR:
            self.logger = PbarLogger(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.logger.update(*args, **kwargs)

    def write(self, message, color=None, mode=None):
        self.logger.write(message, color, mode)


class DefaultLogger(object):

    def __init__(self, total):
        self.total = total
        self.count = 0

    def update(self, n=1, *args, **kwargs):
        pass

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
        self.pbar.write(decorate_message(message, color, mode))


def decorate_message(message, color=None, mode=None):
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
