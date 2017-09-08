"""
This code is a modified copy from the original Guided Policy Search implementation.
The original code can be found here: https://github.com/cbfinn/gps

The following license applies:


COPYRIGHT

All contributions by the University of California:
Copyright (c) 2015, 2016 The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2015, 2016, the respective contributors
All rights reserved.

GPS uses a shared copyright model: each contributor holds copyright over
their contributions to the GPS codebase. The project versioning records all such
contribution and copyright details. If a contributor wants to further mark
their specific copyright on a particular contribution, they should indicate
their copyright solely in the commit message of the change when it is
committed.

LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CONTRIBUTION AGREEMENT

By contributing to the GPS repository through pull-request, comment,
or otherwise, the contributor releases their content to the
license and copyright terms herein.

"""
import logging

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from .policy import LinearGaussianPolicy
from .dynamics import LRDynamics
from .sample import Sample


LOGGER = logging.getLogger(__name__)


class TrajOptiLQG(object):

    def __init__(self,
                 max_itr=50,
                 del0=1e-4,
                 eta=1.0,
                 step_mult=1.0,
                 eta_error_threshold=1e16,
                 min_eta=1e-8,
                 max_eta=1e16,
                 base_kl_step=0.2):
        self.max_itr = max_itr
        self.del0 = del0
        self.eta = eta
        self.step_mult = step_mult
        self.eta_error_threshold = eta_error_threshold
        self.min_eta = min_eta
        self.max_eta = max_eta
        self.base_kl_step = base_kl_step

    def optimize(self, sample_list, prev_traj_distr, dynamics, cost_func):
        min_eta = self.min_eta
        max_eta = self.max_eta

        kl_step = self.base_kl_step * self.step_mult * prev_traj_distr.T
        for itr in range(self.max_itr):
            LOGGER.debug("Iteration %d, KL params: (%.2e, %.2e, %.2e)", itr, self.min_eta, self.eta, self.max_eta)

            # Perform backward pass
            traj_distr = self.backward_pass(sample_list, dynamics, cost_func)

            # Perform forward pass and calculate KL divergence
            mu, sigma = self.forward_pass(traj_distr, dynamics)
            kl_div = self.traj_distr_kl(mu, sigma, traj_distr, prev_traj_distr)

            con = kl_div - kl_step

            # Check if converged
            if self._conv_check(con, kl_step):
                LOGGER.debug("KL: %f / %f, converged in iteration %d", kl_div, kl_step, itr)
                break

            # Choose new eta
            if con < 0:  # eta was too big
                max_eta = self.eta
                geom = np.sqrt(min_eta*max_eta) # Geometric mean
                new_eta = max(geom, 0.1*max_eta)
                LOGGER.debug("KL: %f / %f, eta too big, new eta: %f", kl_div, kl_step, new_eta)
            else:  # eta was too small
                min_eta = self.eta
                geom = np.sqrt(min_eta*max_eta) # Geometric mean
                new_eta = min(geom, 10.0*min_eta)
                LOGGER.debug("KL: %f / %f, eta too small, new eta: %f", kl_div, kl_step, new_eta)

            self.eta = new_eta

        if np.mean(kl_div) > np.mean(kl_step) and not self._conv_check(con, kl_step):
            LOGGER.warning("Final KL divergence after DGD convergence is too high.")

        return traj_distr

    def forward_pass(self, traj_distr, dynamics):
        T, dX, dU = traj_distr.T, traj_distr.dX, traj_distr.dU

        # Constants
        idx_x = slice(dX)

        # Initialize mu and sigma
        mu = np.zeros((T, dX + dU))
        sigma = np.zeros((T, dX + dU, dX + dU))

        # Pull out dynamics
        Fm = dynamics.Fm
        fv = dynamics.fv
        dyn_cov = dynamics.dyn_cov

        for t in range(T):
            sigma[t, :, :] = np.vstack([
                np.hstack([
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)
                ]),
                np.hstack([
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                    traj_distr.pol_covar + traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(traj_distr.K[t, :, :].T)
                ])
            ])
            mu[t, :] = np.hstack([
                mu[t, idx_x],
                traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]
            ])

            if t < T - 1:
                sigma[t+1, idx_x, idx_x] = Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + dyn_cov[t, :, :]
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]

        return mu, sigma

    def backward_pass(self, sample_list, dynamics, cost_func):
        T, dX, dU = sample_list.T, sample_list.dX, sample_list.dU

        # Initialize new trajectory distribution
        traj_distr = LinearGaussianPolicy.init_nans(T, dX, dU)

        # Constants
        idx_x = slice(dX)
        idx_u = slice(dX, dX+dU)

        # Pull out dynamics
        Fm = dynamics.Fm
        fv = dynamics.fv

        # Non-SPD correction terms
        del_ = self.del0
        eta0 = self.eta

        # Run dynamic programming
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric PD

            # Allocate space
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))
            Qtt = np.zeros((T, dX + dU, dX + dU))
            Qt = np.zeros((T, dX + dU))

            cs, fCm, fcv = self.compute_cost(sample_list, cost_func)

            for t in range(T - 1, -1, -1):
                # Add in costs
                Qtt[t, :, :] = fCm[t, :, :]
                Qt[t, :] = fcv[t, :]

                if t < T - 1:
                    Qtt[t, :, :] += Fm[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t, :, :])
                    Qt[t, :] += Fm[t, :, :].T.dot(Vx[t + 1, :] + Vxx[t+1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component
                Qtt[t, :, :] = 0.5 * (Qtt[t, :, :] + Qtt[t, :, :].T)

                # Extract components
                inv_term = Qtt[t, idx_u, idx_u]
                k_term = Qt[t, idx_u]
                K_term = Qtt[t, idx_u, idx_x]

                # Compute Cholesky decomposition of Q function action component
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    LOGGER.debug("LinAlgError: %s", e)
                    fail = True
                    break

                traj_distr.inv_pol_covar[t, :, :] = inv_term
                traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
                )
                traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(
                    traj_distr.pol_covar[t, :, :]
                )

                # Compute mean terms.
                traj_distr.k[t, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, k_term, lower=True)
                )
                traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, K_term, lower=True)
                )

                # Compute value function
                Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + Qtt[t, idx_x, idx_u].dot(traj_distr.K[t, :, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)
                Vx[t, :] = Qt[t, idx_x] + Qtt[t, idx_x, idx_u].dot(traj_distr.k[t, :])

                # Increment eta on non-SPD Q-function
                if fail:
                    old_eta = self.eta
                    self.eta = eta0 + del_
                    LOGGER.debug('Increasing eta: %f -> %f', old_eta, self.eta)
                    del_ *= 2  # Increase del_ exponentially on failure.

                    fail_check = (self.eta >= 1e16)
                    if fail_check:
                        if np.any(np.isnan(Fm) or np.any(np.isnan(fv))):
                            raise ValueError('NaNs encountered in dynamics!')
                        raise ValueError('Failed to find PD solution even for very \
                                large eta (check that dynamics and cost are \
                                reasonably well conditioned)!')

        return traj_distr

    def traj_distr_kl(self, mu, sigma, traj_distr, prev_traj_distr):
        T, dX, dU = traj_distr.T, traj_distr.dX, traj_distr.dU

        # Initialize KL vector
        kl_div = np.zeros(T)

        # Step through trajectory
        for t in range(T):
            # Fetch matrices and vectors from trajectory distributions.
            mu_t = mu[t, :dX]
            sigma_t = sigma[t, :dX, :dX]

            K_prev = prev_traj_distr.K[t, :, :]
            K_new = traj_distr.K[t, :, :]

            k_prev = prev_traj_distr.k[t, :]
            k_new = traj_distr.k[t, :]

            sig_prev = prev_traj_distr.pol_covar[t, :, :]
            sig_new = prev_traj_distr.pol_covar[t, :, :]

            chol_prev = prev_traj_distr.chol_pol_covar[t, :, :]
            chol_new = traj_distr[t, :, :]

            inv_prev = prev_traj_distr.inv_pol_covar[t, :, :]
            inv_new = prev_traj_distr.inv

            logdet_prev = 2 * sum(np.log(np.diag(chol_prev)))
            logdet_new = 2 * sum(np.log(np.diag(chol_new)))

            K_diff, k_diff = K_prev - K_new, k_prev - k_new

            # TODO: How does the KL_div calculation work?
            kl_div[t] = max(
                0,
                0.5 * (logdet_prev - logdet_new - dU +
                       np.sum(np.diag(inv_prev.dot(sig_new))) +
                       k_diff.T.dot(inv_prev).dot(k_diff) +
                       mu_t.T.dot(K_diff.T).dot(inv_prev).dot(K_diff).dot(mu_t) +
                       np.sum(np.diag(K_diff.T.dot(inv_prev).dot(K_diff).dot(sigma_t))) +
                       2 * k_diff.T.dot(inv_prev).dot(K_diff).dot(mu_t))
            )

        return np.sum(kl_div)

    def _conv_check(self, con, kl_step):
        """Function that checks whether dual gradient descent has converged."""
        return abs(con) < 0.1 * kl_step

    def compute_cost(self, sample_list, cost_func):
        T, dX, dU = sample_list.T, sample_list.dX, sample_list.dU
        N = sample_list.len()

        # Allocate space
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX + dU))
        Cm = np.zeros((N, T, dX + dU, dX + dU))
        for n in range(N):
            sample = sample_list.get_samples(n)

            # Get cost
            l, lx, lu, lxx, luu, lux = cost_func(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample
            # TODO: What is this?
            # X = sample.get_X()
            # U = sample.get_U()
            # yhat = np.c_[X, U]
            # rdiff = -yhat
            # rdiff_expand = np.expand_dims(rdiff, axis=2)
            # cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            # cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            # cv[n, :, :] += cv_update

        return cs, np.mean(Cm, 0), np.mean(cv, 0)

    def estimate_cost(self):
        pass

    def _adjust_step_mult(self):
        # TODO: Implement adjust KL step mult
        pass