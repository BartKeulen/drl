import numpy as np


def runge_kutta(fun, x, u, dt):
    """
    Implementation of fourth order Runge-Kutta method for solving differential equations.
    The method solves equations of the following form:

        dx/dt = f(x, u)

    The solution is defined as:

        x(k+1) = x(k) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        with:

        k1 = f(x, u)
        k2 = f(x + k1 * h/2, u)
        k3 = f(x + k2 * h/2, u)
        k4 = f(x + k3 * h, u)


    :param fun: the function to be solved
    :param x: current state
    :param u: current control input
    :param dt: time step
    :return: next state, dx/dt
    """
    k1 = fun(x, u)
    k2 = fun(x + dt / 2. * k1, u)
    k3 = fun(x + dt / 2. * k2, u)
    k4 = fun(x + dt * k3, u)

    xdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6.

    return x + dt * xdot, xdot


def function_derivatives(x, u, func, first=None, second=False):
    """
    Computes the first and second order derivatives of a dynamics function using finite difference.

    Function must be of the form:

        y = f(x, u)


    :param x: current state
    :param u: current control input
    :param func: function to take the derivative from
    :param first: if the first derivative is given it is used for calculating the second derivative
    :param second: boolean if the second derivative has to be calculated
    :return: first and second order derivatives: dydx, dydu, dydxx, dydxu, dyduu
    """

    xi = np.arange(x.shape[1])
    ui = np.arange(u.shape[1]) + x.shape[1]

    # first derivatives if not given
    if first is None:
        xu_func = lambda xu: func(xu[:, xi], xu[:, ui])
        J = finite_difference(xu_func, np.hstack((x, u)))
        dx = J[:, xi]
        du = J[:, ui]
    else:
        dx, du = first

    # Second derivatives if requested
    if second:
        xu_Jfunc = lambda xu: finite_difference(xu_func, xu)
        JJ = finite_difference(xu_Jfunc, np.hstack((x, u)))
        dxx = JJ[:, xi][:, :, xi]
        dxu = JJ[:, xi][:, :, ui]
        duu = JJ[:, ui][:, :, ui]
    else:
        dxx = None
        dxu = None
        duu = None

    return dx, du, dxx, dxu, duu


def finite_difference(fun, x, h=2e-6):
    """
    Computes the derivatives of a function using finite difference.
    The derivative to each input is calculated.

    Function must be of the form:

        y = f(x)

    NOTE: function must be vectorized!


    :param fun: function to take the derivative from
    :param x: input to the function
    :param h: step-size
    :return: derivative dydx
    """

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