import numpy as np


class LinearGaussianPolicy(object):

    def __init__(self, K, k, inv_pol_covar, pol_covar, chol_pol_covar):
        self.T = K.shape[0]
        self.dX = K.shape[2]
        self.dU = K.shape[1]

        self.K = K
        self.k = k
        self.inv_pol_covar = inv_pol_covar  # Quu
        self.pol_covar = pol_covar  # inv(Quu)
        self.chol_pol_covar = chol_pol_covar  # Chol(inv(Quu))

    def act(self, x, t, noise=None):
        u = self.K[t].dot(x) + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def nans_like(self):
        traj_distr = LinearGaussianPolicy(
            np.zeros_like(self.K), np.zeros_like(self.k), np.zeros_like(self.inv_pol_covar),
            np.zeros_like(self.pol_covar), np.zeros_like(self.chol_pol_covar)
        )

        traj_distr.K.fill(np.nan)
        traj_distr.k.fill(np.nan)
        traj_distr.inv_pol_covar.fill(np.nan)
        traj_distr.pol_covar.fill(np.nan)
        traj_distr.chol_pol_covar.fill(np.nan)

        return traj_distr

    @staticmethod
    def init_nans(T, dX, dU):
        K = np.zeros((T, dU, dX))
        k = np.zeros((T, dU))
        inv_pol_covar = np.zeros((T, dU, dU))
        pol_covar = np.zeros((T, dU, dU))
        chol_pol_covar = np.zeros((T, dU, dU))

        traj_distr = LinearGaussianPolicy(K, k, inv_pol_covar, pol_covar, chol_pol_covar)

        traj_distr.K.fill(np.nan)
        traj_distr.k.fill(np.nan)
        traj_distr.inv_pol_covar.fill(np.nan)
        traj_distr.pol_covar.fill(np.nan)
        traj_distr.chol_pol_covar.fill(np.nan)

        return traj_distr


def init_lqr(init_traj, dynamics, cost_func):
    T, dX, dU = init_traj.T, init_traj.dX, init_traj.dU
    X, U = init_traj.get_X(), init_traj.get_U()

    x0 = X[0,:]

    idx_x = slice(dX)
    idx_u = slice(dX, dX + dU)

    init_acc = np.zeros(dU)
    init_gains = np.ones(dU)

    # Set up simple linear dynamics model.
    Fm, Fv = dynamics.Fm, dynamics.Fv

    l, lx, lu, lxx, luu, lux = cost_func(init_traj)