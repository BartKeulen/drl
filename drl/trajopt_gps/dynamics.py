import numpy as np


class LRDynamics(object):

    def __init__(self,
                 regularization=1e-6):
        self.Fm = None
        self.fv = None
        self.dyn_cov = None
        self.regularization=regularization

    def fit(self, X, U):
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX + dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX + dU)
        ip = slice(dX + dU, dX + dU + dX)
        # Fit dynamics wih least squares regression.
        for t in range(T - 1):
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t + 1, :]]
            xux_mean = np.mean(xux, axis=0)
            empsig = (xux - xux_mean).T.dot(xux - xux_mean) / N
            sigma = 0.5 * (empsig + empsig.T)
            sigma[it, it] += self.regularization

            Fm = np.linalg.solve(sigma[it, it], sigma[it, ip]).T
            fv = xux_mean[ip] - Fm.dot(xux_mean[it])

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv

            dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)
            self.dyn_covar[t, :, :] = 0.5 * (dyn_covar + dyn_covar.T)

        return self.Fm, self.fv, self.dyn_covar


class Dynamics(object):

    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.Fm = None
        self.fv = None
        self.dyn_cov = None

    def fit(self, X, U):
        T = X.shape[1]

        Fm, fv, dyn_covar = self.dynamics()

        Fm = np.expand_dims(Fm, 0)
        self.Fm = np.repeat(Fm, T, 0)

        fv = np.expand_dims(fv, 0)
        self.fv = np.repeat(fv, T, 0)

        dyn_covar = np.expand_dims(dyn_covar, 0)
        self.dyn_cov = np.repeat(dyn_covar, T, 0)

        return self.Fm, self.fv, self.dyn_cov
