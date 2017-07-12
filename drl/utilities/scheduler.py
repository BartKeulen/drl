from drl.utilities import Variable, Constant


class Scheduler(object):
    """
    Parent class for implementing a scheduler.
    Linear and Exponential scheduler
    """

    def __init__(self, variable, start, end):
        if not isinstance(variable, Variable):
            self.variable = Constant(variable)
        else:
            self.variable = variable
        self.start = start
        self.end = end

        self.count = 0
        self.scale = 1.

    def _sample(self):
        if self.scale < 0.:
            self.scale = 0.

        return self.variable.sample() * self.scale

    def update(self):
        self.count += 1

    def reset(self):
        self.count = 0
        self.scale = 1.


class ConstantScheduler(Scheduler):

    def __init__(self, constant, *args, **kwargs):
        super(ConstantScheduler, self).__init__(*args, **kwargs)
        self.constant = constant

    def sample(self):
        return self.variable.sample() * self.constant


class LinearScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super(LinearScheduler, self).__init__(*args, **kwargs)

    def sample(self):
        if self.end > self.count > self.start:
            self.scale -= 1. / (self.end - self.start)
        elif self.count >= self.end:
            self.scale = 0.

        return super(LinearScheduler, self)._sample()


class ExponentialScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super(ExponentialScheduler, self).__init__(*args, **kwargs)

    def sample(self):
        if self.end > self.count > self.start:
            self.scale = 1 / (1 + self.count - self.start)
        elif self.count >= self.end:
            self.scale = 0.

        return super(ExponentialScheduler, self)._sample()