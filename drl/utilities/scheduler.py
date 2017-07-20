from abc import ABCMeta, abstractmethod


class Variable(metaclass=ABCMeta):
    """
    Used for implementing variables that can be used by different sub processes like the Scheduler.
    """

    @abstractmethod
    def sample(self):
        pass


class Constant(Variable):
    """
    Implementation of a constant Variable.
    """

    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


class Scheduler(object):
    """
    Parent class for implementing a scheduler. A scheduler is used to decrease the value of a Variable over time,
    the Variable is scaled from 1 to 0 between a start and end point.
    """

    def __init__(self, variable, start, end):
        """
        Constructs the scheduler.

        :param variable: Can be a Variable or value.
        :param start: Time-step when the Scheduler has to start
        :param end: Time-step when the Scheduler has to finish
        """
        if not isinstance(variable, Variable):
            self.variable = Constant(variable)
        else:
            self.variable = variable
        self.start = start
        self.end = end

        self.count = 0
        self.scale = 1.

    @abstractmethod
    def sample(self):
        pass

    def _sample(self):
        """
        Scales the Variable value and ensures the scaling does not go below zero.

        :return: scaled Variable value
        """
        if self.scale < 0.:
            self.scale = 0.

        return self.variable.sample() * self.scale

    def update(self):
        """
        Increases the count by one.
        """
        self.count += 1

    def reset(self):
        """
        Resets the scheduler
        """
        self.count = 0
        self.scale = 1.


class LinearScheduler(Scheduler):
    """
    Linear scheduler linearly decays the value of the Variable from 1 to 0 between start and end.
    """

    def __init__(self, *args, **kwargs):
        super(LinearScheduler, self).__init__(*args, **kwargs)

    def sample(self):
        """
        Samples the value of the Variable at the current step.
        :return: scaled Variable value
        """
        if self.end > self.count > self.start:
            self.scale -= 1. / (self.end - self.start)
        elif self.count >= self.end:
            self.scale = 0.

        return super(LinearScheduler, self)._sample()


class ExponentialScheduler(Scheduler):
    """
    Exponential scheduler exponentially decays the value of the Variable form 1 to 0 between start and end.
    """

    def __init__(self, *args, **kwargs):
        super(ExponentialScheduler, self).__init__(*args, **kwargs)

    def sample(self):
        """
        Samples the value of the Variable at the current step.
        :return: scaled Variable value
        """
        if self.end > self.count > self.start:
            self.scale = 1 / (1 + self.count - self.start)
        elif self.count >= self.end:
            self.scale = 0.

        return super(ExponentialScheduler, self)._sample()