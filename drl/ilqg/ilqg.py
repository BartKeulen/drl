import numpy as np

EPSILON = 1e-6


class iLQG(object):

    def __init__(self, env):
        self.env = env

    def forward_pass(self, pi):
        T = len(pi)

        # Roll-out trajectory
        trajectory = []
        x = self.env.reset()
        for i in range(T):
            u = pi[i]
            trajectory.append((x, u))
            x = self.env.step(u)
        trajectory.append((x, 0.))



    def cost(self, t):
        x, u = t
        l = 0.01 * u**2
        lx = np.zeros([2, 1])
        lu = 0.02 * u
        lxx = np.zeros([2, 2])
        luu = 0.02
        lux = np.zeros(2)
        return l, lx, lu, lxx, luu, lux

    def final_cost(self, t):
        x, u = t
        l = (1 - np.cos(x[0]))**2 + 0.1 * x[1]**2 + 0.01 * u**2
        lx = np.array([2.*(1. - np.cos(x[0]))*np.sin(x[0]), 0.2*x[1]]).reshape([2, 1])
        lu = 0.02 * u
        lxx = np.array([4.*np.sin(x[0]/2.)**2 * (2*np.cos(x[0]) + 1), 0., 0., 0.2]).reshape([2, 2])
        luu = 0.02
        lux = 0.
        return l, lx, lu, lxx, luu, lux

    def dynamics_derivatives(self, t):
        x, u = t

        dfdx = np.zeros(len(x))
        for i in range(len(x)):
            x_plus = x
            x_plus[i] += EPSILON
            x_min = x
            x_min[i] -= EPSILON
            dfdx[i] = (self.calc_next_state(x_plus, u) - self.calc_next_state(x_min, u)) / (2. * EPSILON)

        dfdu = np.zeros(len(u))
        for i in range(len(u)):
            u_plus = u
            u_plus[i] += EPSILON
            u_min = u
            u_min[i] -= EPSILON
            dfdu[i] = (self.calc_next_state(x, u_plus) - self.calc_next_state(x, u_min)) / (2. * EPSILON)

        return dfdx.reshape([len(x), 1]), dfdu.reshape([len(u), 1])

    def calc_next_state(self, x, u):
        th, thdot = x

        max_torque = self.env.action_space.high[0]
        state_bounds = self.env.observation_space.high[0]

        g = 10.
        m = 1.
        l = 1.
        dt = 0.05

        u = np.clip(u, -max_torque, max_torque)
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt

        newthdot = np.clip(newthdot, -state_bounds[1], state_bounds[1])

        return np.array([np.cos(newth), np.sin(newth), newthdot])
