import numpy as np
from abc import ABCMeta, abstractmethod


class Exploration(metaclass=ABCMeta):
    """
    Base class for exploration noise.

    Exploration noise is added to the action in order to explore instead of exploiting the current policy.
    """

    def __init__(self, action_dim):
        """
        Constructs an Exploration object.

        :param action_dim:
        """
        self.action_dim = action_dim

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

    def __init__(self, action_dim, constant=0):
        """
        Constructs a ConstantNoise object.

        :param action_dim:
        :param constant: constant value to return (standard 0)
        """
        super(ConstantNoise, self).__init__(action_dim)
        self.constant = constant

    def sample(self):
        """
        :return: constant
        """
        return self.constant


class WhiteNoise(Exploration):
    """
    White Noise class generates random white noise.
    """

    def __init__(self, action_dim, mu, sigma):
        """
        Constructs a WhiteNoise object.

        :param action_dim:
        :param mu: mean of the noise
        :param sigma: variance of the noise
        """
        super(WhiteNoise, self).__init__(action_dim)
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        """
        :return: random noise with mean mu and variance sigma
        """
        return np.random.randn(self.action_dim)*self.sigma + self.mu


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

    def __init__(self, action_dim, mu, sigma, theta):
        """
        Construct an OrnSteinUhlenbeckNoise object.

        :param action_dim:
        :param mu: mean of the noise signal
        :param sigma: variance
        :param theta: dependency on staying close to the mean
        """
        super(OrnSteinUhlenbeckNoise, self).__init__(action_dim)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.reset()

    def sample(self):
        """
        Saves the new noise state for next iteration.

        :return: next noise state
        """
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        return self.state

    def reset(self):
        """
        Resets the noise signal to zero mean and variance sigma
        """
        self.state = np.random.randn(self.action_dim)*self.sigma


class NoiseDecay(metaclass=ABCMeta):
    """
    Parent class for implementing noise decay.
    """

    def __init__(self, exploration, decay_start, decay_end):
        """
        Construct a NoiseDecay object.

        :param exploration: Exploration object, e.g. WhiteNoise
        :param decay_start: Time-step the decay starts
        :param decay_end: Time-step the noise is zero
        """
        self.exploration = exploration
        self.decay_start = decay_start
        self.decay_end = decay_end
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

    def __init__(self, exploration,  decay_start, decay_end):
        """
        Constructs LinearDecay object.

        :param exploration: Exploration object. e.g. WhiteNoise
        :param decay_start: Time-step the decay starts
        :param decay_end: Time-step the decay ends
        """
        super(LinearDecay, self).__init__(exploration, decay_start, decay_end)
        self.width = decay_end - decay_start
        self.scaling = 1.

    def sample(self):
        """
        :return: scaled noise value
        """
        if self.step < self.decay_start:
            return self.exploration.sample()
        elif self.step >= self.decay_end:
            return 0.

        return self.exploration.sample() * (1 - self.step/self.width)


class ExponentialDecay(NoiseDecay):
    """
    Noise decays exponentially by dividing the signal with (1 + step) between start and end.
    """

    def __init__(self, exploration, decay_start, decay_end):
        """
        Constructs ExponentialDecay object.

        :param exploration: Exploration object. e.g. WhiteNoise
        :param decay_start: Time-step the decay starts
        :param decay_end: Time-step the decay ends
        """
        super(ExponentialDecay, self).__init__(exploration, decay_start, decay_end)

    def sample(self):
        """
        :return: scaled noise value
        """
        if self.step < self.decay_start:
            return self.exploration.sample()
        elif self.step >= self.decay_end:
            return 0.

        return self.exploration.sample() / (1 + self.step)