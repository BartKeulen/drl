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