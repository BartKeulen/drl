import numpy as np
from abc import ABCMeta, abstractmethod
from drl.utilities import Variable


class ExplorationStrategy(Variable, metaclass=ABCMeta):
    """
    Base class for explorationstrategy noise.

    Exploration noise is added to the action in order to explore instead of exploiting the current policy.
    """

    def __init__(self, action_dim, scale=1.):
        """
        Constructs an Exploration object.

        :param action_dim:
        """
        self.action_dim = action_dim
        self.scale = scale

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


class ConstantStrategy(ExplorationStrategy):
    """
    Constant Noise class return a constant value as noise.
    """

    def __init__(self, action_dim, mu=1., scale=1.):
        """
        Constructs a ConstantNoise object.

        :param action_dim:
        :param constant: constant value to return (standard 0)
        """
        super(ConstantStrategy, self).__init__(action_dim, scale)
        self.mu = mu

    def sample(self):
        """
        :return: constant
        """
        return np.ones(self.action_dim) * self.mu * self.scale


class WhiteNoiseStrategy(ExplorationStrategy):
    """
    White Noise class generates random white noise.
    """

    def __init__(self, action_dim, mu=0., sigma=1., scale=1.):
        """
        Constructs a WhiteNoise object.

        :param action_dim:
        :param mu: mean of the noise
        :param scale: variance of the noise
        """
        super(WhiteNoiseStrategy, self).__init__(action_dim, scale)
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        """
        :return: random noise with mean mu and variance sigma
        """
        state = np.random.randn(self.action_dim)*self.sigma + self.mu
        return state * self.scale


class OrnSteinUhlenbeckStrategy(ExplorationStrategy):
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

    def __init__(self, action_dim, mu=0., sigma=0.2, theta=0.15, scale=1.):
        """
        Construct an OrnSteinUhlenbeckNoise object.

        :param action_dim:
        :param mu: mean of the noise signal
        :param sigma: variance
        :param theta: dependency on staying close to the mean
        """
        super(OrnSteinUhlenbeckStrategy, self).__init__(action_dim, scale)
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
        return self.state * self.scale

    def reset(self):
        """
        Resets the noise signal to zero mean and variance sigma
        """
        self.state = np.random.randn(self.action_dim)*self.sigma * self.scale
