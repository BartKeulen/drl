import numpy as np


class Exploration(object):

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def get_noise(self):
        return None

    def reset(self):
        pass

    def increase(self):
        pass


class ConstantNoise(Exploration):

    def __init__(self, action_dim, constant):
        super(ConstantNoise, self).__init__(action_dim)
        self.constant = constant

    def get_noise(self):
        return self.constant


class WhiteNoise(Exploration):

    def __init__(self, action_dim, mu, sigma):
        super(WhiteNoise, self).__init__(action_dim)
        self.mu = mu
        self.sigma = sigma

    def get_noise(self):
        return np.random.randn(self.action_dim)*self.sigma + self.mu


class OrnSteinUhlenbeckNoise(Exploration):

    def __init__(self, action_dim, mu, sigma, theta):
        super(OrnSteinUhlenbeckNoise, self).__init__(action_dim)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.reset()

    def get_noise(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        return self.state

    def reset(self):
        self.state = np.random.randn(self.action_dim)*self.sigma


class NoiseDecay(object):

    def __init__(self, exploration, decay_start, decay_end):
        self.exploration = exploration
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.step = 0

    def increase(self):
        self.step += 1

    def get_noise(self):
        return None

    def reset(self):
        self.exploration.reset()


class LinearDecay(NoiseDecay):

    def __init__(self, exploration,  decay_start, decay_end):
        super(LinearDecay, self).__init__(exploration, decay_start, decay_end)

    def get_noise(self):
        if self.step < self.decay_start:
            return self.exploration.get_noise()
        elif self.step >= self.decay_end:
            return 0.

        return self.exploration.get_noise() / (1 + self.step)


class ExponentialDecay(NoiseDecay):

    def __init__(self, exploration, decay_start, decay_end):
        super(ExponentialDecay, self).__init__(exploration, decay_start, decay_end)

    def get_noise(self):
        if self.step < self.decay_start:
            return self.exploration.get_noise()
        elif self.step >= self.decay_end:
            return 0.

