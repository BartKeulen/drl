import numpy as np


class LinearGaussianPolicy(object):

    def __init__(self, T, dX, dU):
        self.T = T
        self.dX = dX
        self.dU = dU

        self.K = None
        self.k = None
        self.sigma = None
        self.alpha = 1.0
        self.x_prev = np.zeros((self.T, self.dX))
        self.u_prev = np.zeros((self.T-1, self.dU))

    def fit(self, K, k, sigma, x_prev=None, u_prev=None, alpha=1.0):
        assert K.shape == (self.T, self.dU, self.dX)
        assert k.shape == (self.T, self.dU)
        assert sigma.shape == (self.T, self.dU, self.dU)

        self.K = K
        self.k = k
        self.sigma = sigma

        if x_prev is not None and u_prev is not None:
            assert x_prev.shape == self.x_prev.shape
            assert u_prev.shape == self.u_prev.shape

            self.x_prev = x_prev
            self.u_prev = u_prev

        self.alpha = alpha

    def init_random(self):
        self.K = np.random.randn(self.T, self.dU, self.dX)
        self.k = np.random.randn(self.T, self.dU)
        self.sigma = np.tile(np.eye(self.dU)[np.newaxis], (self.T, 1, 1))

    def act(self, t, x, noise=None):
        if self.K is None or self.k is None or self.sigma is None:
            raise RuntimeError("Policy not initialized.")

        u = self.u_prev[t, :] + self.alpha * self.k[t] + self.K[t].dot(x - self.x_prev[t, :])
        if noise is not None:
            u += np.dot(self.sigma[t], noise)
        return u

    @staticmethod
    def init_nans(T, dX, dU):
        K = np.zeros((T, dU, dX)).fill(np.nan)
        k = np.zeros((T, dU)).fill(np.nan)
        sigma = np.zeros((T, dU, dU)).fill(np.nan)

        traj_distr = LinearGaussianPolicy(T, dX, dU)
        traj_distr.fit(K, k, sigma)

        return traj_distr

    def get(self, T):
        assert T <= self.T
        K = self.K[:T, :, :]
        k = self.k[:T, :]
        sigma = self.sigma[:T, :, :]
        x_prev = self.x_prev[:T, :]
        u_prev = self.u_prev[:T, :]
        policy = LinearGaussianPolicy(T, self.dX, self.dU)
        policy.fit(K, k, sigma, x_prev, u_prev)
        return policy

    def concatenate(self, policy, pos=-1):
        assert self.dX == policy.dX
        assert self.dU == policy.dU

        T = self.T + policy.T

        new_policy = LinearGaussianPolicy(T, self.dX, self.dU)
        if pos == -1:
            before = policy
            after = self
        elif pos == 1:
            before = self
            after = policy
        else:
            raise Exception("Position must be -1 or 1 for adding policy before or after the current policy")

        K = np.concatenate((before.K, after.K), axis=0)
        k = np.concatenate((before.k, after.k), axis=0)
        sigma = np.concatenate((before.sigma, after.sigma), axis=0)

        new_policy.fit(K, k, sigma)
        return new_policy