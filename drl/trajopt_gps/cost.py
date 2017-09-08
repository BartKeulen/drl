import numpy as np

from drl.utilities.numerical import function_derivatives


def _cost_func(x, u, x_goal, wp=None, wv=None, wu=None):
    T = x.shape[0]
    l = np.zeros(T)
    if wp is not None:
        d = x[:, :2] - x_goal
        l[:] += 0.5 * wp * np.sum(d * d, axis=1)
    if wv is not None:
        l[:] += 0.5 * wv * np.sum(x[:, 2:] * x[:, 2:], axis=1)
    if wu is not None and not np.any(np.isnan(u)):
        l[:] += 0.5 * wu * np.sum(u * u, axis=1)

    return l


def cost_func(sample, x_goal, wp=None, wv=None, wu=None):
    T = sample.T
    dX = sample.dX
    dU = sample.dU

    X = sample.get_X()
    U = sample.get_U()

    l = np.zeros(T)
    lx = np.zeros((T, dX))
    lu = np.zeros((T, dU))
    lxx = np.zeros((T, dX, dX))
    luu = np.zeros((T, dU, dU))
    lux = np.zeros((T, dU, dX))

    c = lambda x_in, u_in: _cost_func(x_in, u_in, x_goal, wp, wv, wu)
    for t in range(T):
        x = np.expand_dims(X[t, :], 0)
        if t < T:
            u = np.expand_dims(U[t, :], 0)
        else:
            u = np.zeros((1, dU))
            u.fill(np.nan)

        l[t] = _cost_func(x, u, x_goal, wp, wv, wu)
        lx[t, :], lu[t, :], lxx[t, :], luu[t, :, :], lux[t, :, :] = function_derivatives(x, u, c, True)

    return l, lx, lu, lxx, luu, lux







