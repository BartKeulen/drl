from .arm import Arm
import numpy as np


class TwoLinkArm(Arm):
    """
    Two link robotic arm object. Child object of 'Arm' object.
    """

    def __init__(self, m1=1., l1=1., m2=1., l2=1., g=9.81, dt=0.05, wp=10., wv=1., wu=0.001, action_high=None, velocity_high=None):
        """
        Construct a new 'TwoLinkArm' object.

        :param m1: mass of link 1
        :param l1: length of link 1
        :param m2: mass of link 2
        :param l2: length of link 2
        :param g: gravity
        :param dt: time-step
        :param wp: weight on position error
        :param wv: weight on velocity
        :param wu: weight on control input
        :param action_high: action limit (assumed lower limit is the same as high limit)
        :param velocity_high: velocity limit (assumed lower limit is the same as high limit)
        """
        super(TwoLinkArm, self).__init__(2, g=g, dt=dt, wp=wp, wv=wv, wu=wu, action_high=action_high, velocity_high=velocity_high)
        self.params += (m1, l1, m2, l2)

        self.B = np.zeros((self.dof, self.dof))
        self.C = np.zeros((self.dof, 1))
        self.G = np.zeros((self.dof, 1))

    def cost_func(self, q, u):
        """
        Updated cost function, also returns first and second derivatives of the cost function wrt the state and action.

        :param q: state
        :param u: control input
        :return: cost, dcdx, dcdu, dcdxx, dcduu, dcdxu
        """
        final = np.isnan(u)
        u[final] = 0

        lx = np.zeros(self.state_dim)
        lxx = np.zeros((self.state_dim, self.state_dim))
        lxu = np.zeros((self.state_dim, self.action_dim))

        cu = self.wu * np.sum(u * u)
        lu = self.wu * u
        luu = np.eye(self.action_dim) * self.wu

        if final.any():
            d = self._distance(q)
            cp = self.wp * d
            cv = self.wv * np.sum(q[self.dof:] * q[self.dof:])

            _, m1, l1, m2, l2 = self.params
            pos = self.to_cartesian(q)
            x = pos[-2]
            y = pos[-1]
            goal = self.to_cartesian(self.goal)
            xs = goal[-2]
            ys = goal[-1]

            lx[0] = (-xs*y + ys*x)/d
            lx[1] = (l2*(-l1*np.sin(q[1]) - l2*xs*np.cos(q[0]+q[1]) + l2*ys*np.sin(q[0]+q[1])))/d
            lx[:self.dof] *= self.wp
            lx[self.dof:] = 2. * self.wv * q[self.dof:]

            lxx[0,0] = (((l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1]) - xs)**2 + (l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]) - ys)**2)*(l1*xs*np.sin(q[0]) + l1*ys*np.cos(q[0]) + l2*xs*np.sin(q[0] + q[1]) + l2*ys*np.cos(q[0] + q[1])) - (l1*xs*np.cos(q[0]) - l1*ys*np.sin(q[0]) + l2*xs*np.cos(q[0] + q[1]) - l2*ys*np.sin(q[0] + q[1]))**2)/((l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1]) - xs)**2 + (l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]) - ys)**2)**(3/2)
            lxx[0,1] = l2*((xs*np.sin(q[0] + q[1]) + ys*np.cos(q[0] + q[1]))*((l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1]) - xs)**2 + (l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]) - ys)**2) + (-l1*np.sin(q[1]) - xs*np.cos(q[0] + q[1]) + ys*np.sin(q[0] + q[1]))*(l1*xs*np.cos(q[0]) - l1*ys*np.sin(q[0]) + l2*xs*np.cos(q[0] + q[1]) - l2*ys*np.sin(q[0] + q[1])))/((l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1]) - xs)**2 + (l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]) - ys)**2)**(3/2)
            lxx[1,0] = l2*((xs*np.sin(q[0] + q[1]) + ys*np.cos(q[0] + q[1]))*((l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1]) - xs)**2 + (l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]) - ys)**2) + (-l1*np.sin(q[1]) - xs*np.cos(q[0] + q[1]) + ys*np.sin(q[0] + q[1]))*(l1*xs*np.cos(q[0]) - l1*ys*np.sin(q[0]) + l2*xs*np.cos(q[0] + q[1]) - l2*ys*np.sin(q[0] + q[1])))/((l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1]) - xs)**2 + (l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]) - ys)**2)**(3/2)
            lxx[1,1] = l2*(-l2*(l1*np.sin(q[1]) + xs*np.cos(q[0] + q[1]) - ys*np.sin(q[0] + q[1]))**2 + ((l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1]) - xs)**2 + (l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]) - ys)**2)*(-l1*np.cos(q[1]) + xs*np.sin(q[0] + q[1]) + ys*np.cos(q[0] + q[1])))/((l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1]) - xs)**2 + (l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]) - ys)**2)**(3/2)
            lxx[:self.dof,:self.dof] *= self.wp
            lxx[self.dof:,self.dof:] = 2. * self.wv * np.eye(self.dof)
        else:
            cp = 0
            cv = 0

        c = cu + cp + cv

        return c, lx, lu, lxx, luu, lxu

    def _eom(self, q, u):
        """
        Equations of motion for a two link robotic arm.

        Dynamics equation is:

            1/dt(dtheta/dt) = inv(B(theta))*(-C(dtheta/dt, theta) - G(theta) + F)

            where:

            F = B(theta)*u

        Rewrote to first order differential equation using following state definition

            q = [theta1, theta2, dtheta1/dt, dtheta2/dt]

            dq/dt = [q[2], q[3], 1/dt(dtheta1/dt), 1/dt(dtheta2/dt)]

        :param q: state
        :param u: control input
        :return: dq/dt
        """
        g, m1, l1, m2, l2 = self.params
        q1, q2, q3, q4 = q

        b11 = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * np.cos(q2)
        b12 = m2 * l2 ** 2 + m2 * l1 * l2 * np.cos(q2)
        b21 = m2 * l2 ** 2 + m2 * l1 * l2 * np.cos(q2)
        b22 = m2 * l2 ** 2
        self.B[0, 0] = b11
        self.B[0, 1] = b12
        self.B[1, 0] = b21
        self.B[1, 1] = b22

        c1 = -m2 * l1 * l2 * np.sin(q2) * (2 * q3 * q4 + q4 ** 2)
        c2 = -m2 * l1 * l2 * np.sin(q2) * q3 * q4
        self.C[0, 0] = c1
        self.C[1, 0] = c2

        g1 = -(m1 + m2) * g * l1 * np.sin(q1) - m2 * g * l2 * np.sin(q1 + q2)
        g2 = -m2 * g * l2 * np.sin(q1 + q2)
        self.G[0, 0] = g1
        self.G[1, 0] = g2

        F = np.dot(self.B, u[np.newaxis].T)

        qdotdot = np.dot(np.linalg.inv(self.B), -self.C - self.G + F)

        qdot = np.concatenate(([q3, q4], qdotdot.T[0]))

        return qdot