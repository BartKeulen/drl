import numpy as np


def runge_kutta(fun, x, u, dt):
    k1 = fun(x, u)
    k2 = fun(x + dt / 2. * k1, u)
    k3 = fun(x + dt / 2. * k2, u)
    k4 = fun(x + dt * k3, u)

    xdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6.

    return x + dt * xdot, xdot


def finite_difference(fun, x, h=2e-6):
    # simple finite-difference derivatives
    # assumes the function fun() is vectorized

    K, n = x.shape
    H = np.vstack((-h * np.eye(n), h * np.eye(n)))
    X = x[:, None, :] + H[None, :, :]
    Y = []
    for i in range(K):
        Y.append(fun(X[i]))
    Y = np.array(Y)
    D = (Y[:, n:] - Y[:, 0:n])
    J = D / h / 2
    return J