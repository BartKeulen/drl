import numpy as np
from abc import ABCMeta, abstractmethod
from drl.utilities import Variable

options = {
    'mu': 0.,
    'sigma': 0.2,
    'theta': 0.15,
    'start': 250,
    'end': 300
}


class Exploration(Variable, metaclass=ABCMeta):
    """
    Base class for exploration noise.

    Exploration noise is added to the action in order to explore instead of exploiting the current policy.
    """

    def __init__(self, action_dim, scale=1., options_in=None):
        """
        Constructs an Exploration object.

        :param action_dim:
        """
        self.action_dim = action_dim
        self.scale = scale
        if options_in is not None:
            options.update(options_in)

    @abstractmethod
    def sample(self):
        """
        Samples a noise value.
        """
        pass

    def reset(self):
        """
        Resets the noise generator.
        """
        pass


class ConstantNoise(Exploration):
    """
    Constant Noise class return a constant value as noise.
    """

    def __init__(self, action_dim, scale=1., options_in=None):
        """
        Constructs a ConstantNoise object.

        :param action_dim:
        :param constant: constant value to return (standard 0)
        """
        super(ConstantNoise, self).__init__(action_dim, scale, options_in)

    def sample(self):
        """
        :return: constant
        """
        return np.ones(self.action_dim) * options['mu'] * self.scale


class WhiteNoise(Exploration):
    """
    White Noise class generates random white noise.
    """

    def __init__(self, action_dim, scale=1., options_in=None):
        """
        Constructs a WhiteNoise object.

        :param action_dim:
        :param mu: mean of the noise
        :param scale: variance of the noise
        """
        super(WhiteNoise, self).__init__(action_dim, scale, options_in)

    def sample(self):
        """
        :return: random noise with mean mu and variance sigma
        """
        state = np.random.randn(self.action_dim)*options['sigma'] + options['mu']
        return state * self.scale


class OrnSteinUhlenbeckNoise(Exploration):
    """
    Ornstein Uhlenbeck Noise class generates temporary correlated noise. This is mainly used in control applications.

    The noise is generated according to:
        x[k+1] = x[k] + theta * (mu - x[k]) + sigma * N(0,1)

        With:
            - x[k]: noise at time-step k
            - theta: dependency on staying close to the mean
            - mu: mean of the noise signal
            - sigma: variance
    """

    def __init__(self, action_dim, scale=1., options_in=None):
        """
        Construct an OrnSteinUhlenbeckNoise object.

        :param action_dim:
        :param mu: mean of the noise signal
        :param sigma: variance
        :param theta: dependency on staying close to the mean
        """
        super(OrnSteinUhlenbeckNoise, self).__init__(action_dim, scale, options_in)
        self.reset()

    def sample(self):
        """
        Saves the new noise state for next iteration.

        :return: next noise state
        """
        self.state += options['theta'] * (options['mu'] - self.state) + options['sigma'] * np.random.randn(self.action_dim)
        return self.state * self.scale

    def reset(self):
        """
        Resets the noise signal to zero mean and variance sigma
        """
        self.state = np.random.randn(self.action_dim)*options['sigma'] * self.scale


class NoiseDecay(metaclass=ABCMeta):
    """
    Parent class for implementing noise decay.
    """

    def __init__(self, exploration, options_in=None):
        """
        Construct a NoiseDecay object.

        :param exploration: Exploration object, e.g. WhiteNoise
        :param decay_start: Time-step the decay starts
        :param decay_end: Time-step the noise is zero
        """
        self.exploration = exploration
        if options_in is not None:
            options.update(options_in)
        self.step = 0

    @abstractmethod
    def sample(self):
        """
        Samples the noise signal
        """
        pass

    def reset(self):
        """
        Reset the Exploration object and increase step by one.
        Step is usually equal to the episode.
        """
        self.exploration.reset()
        self.step += 1


class LinearDecay(NoiseDecay):
    """
    Noise decays linear by multiplying the noise signal with a scalar that linearly decays from 1 to 0 between
    start and end.
    """

    def __init__(self, exploration,  options_in=None):
        """
        Constructs LinearDecay object.

        :param exploration: Exploration object. e.g. WhiteNoise
        :param decay_start: Time-step the decay starts
        :param decay_end: Time-step the decay ends
        """
        super(LinearDecay, self).__init__(exploration, options_in)
        self.width = options['end'] - options['start']
        self.scaling = 1.

    def sample(self):
        """
        :return: scaled noise value
        """
        if self.step < options['start']:
            return self.exploration.sample()
        elif self.step >= options['end']:
            return 0.

        return self.exploration.sample() * (1 - self.step/self.width)


class ExponentialDecay(NoiseDecay):
    """
    Noise decays exponentially by dividing the signal with (1 + step) between start and end.
    """

    def __init__(self, exploration, options_in=None):
        """
        Constructs ExponentialDecay object.

        :param exploration: Exploration object. e.g. WhiteNoise
        :param decay_start: Time-step the decay starts
        :param decay_end: Time-step the decay ends
        """
        super(ExponentialDecay, self).__init__(exploration, options_in)

    def sample(self):
        """
        :return: scaled noise value
        """
        if self.step < options['start']:
            return self.exploration.sample()
        elif self.step >= options['end']:
            return 0.

        return self.exploration.sample() / (1 + self.step)