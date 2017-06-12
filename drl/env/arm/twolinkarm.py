from .arm import Arm
import numpy as np


class TwoLinkArm(Arm):
    """
    Two link robotic arm object. Child object of 'Arm' object.
    """

    def __init__(self, m1=1., l1=1., m2=1., l2=1., g=9.81, dt=0.05, action_high=None, velocity_high=None):
        """
        Construct a new 'TwoLinkArm' object.

        :param m1: mass of link 1
        :param l1: length of link 1
        :param m2: mass of link 2
        :param l2: length of link 2
        :param g: gravity
        :param dt: time-step
        :param action_high: action limit (assumed lower limit is the same as high limit)
        :param velocity_high: velocity limit (assumed lower limit is the same as high limit)
        """
        super(TwoLinkArm, self).__init__(2, g=g, dt=dt, action_high=action_high, velocity_high=velocity_high)
        self.params += (m1, l1, m2, l2)

        self.B = np.zeros((self.dof, self.dof))
        self.C = np.zeros((self.dof, 1))
        self.G = np.zeros((self.dof, 1))

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