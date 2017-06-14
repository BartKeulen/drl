from .arm import Arm
import numpy as np


class ThreeLinkArm(Arm):
    """
    Three link robotic arm object. Child object of 'Arm' object.
    """

    def __init__(self, m1=1., l1=1., m2=1., l2=1., m3=1., l3=3., g=9.81, dt=0.05, wp=10., wv=1., wu=0.001, action_high=None, velocity_high=None):
        """
        Construct a new 'ThreeLinkArm' object

        :param m1: mass of link 1
        :param l1: length of link 1
        :param m2: mass of link 2
        :param l2: length of link 2
        :param m3: mass of link 3
        :param l3: length of link 3
        :param g: gravity
        :param dt: time-step
        :param wp: weight on position error
        :param wv: weight on velocity
        :param wu: weight on control input
        :param action_high: action limit (assumed lower limit is the same as high limit)
        :param velocity_high: velocity limit (assumed lower limit is the same as high limit)
        """
        super(ThreeLinkArm, self).__init__(3, g=g, dt=dt, wp=wp, wv=wv, wu=wu, action_high=action_high, velocity_high=velocity_high)
        self.params += (m1, l1, m2, l2, m3, l3)

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

            dq/dt = [q[3], q[4], q[5], 1/dt(dtheta1/dt), 1/dt(dtheta2/dt), 1/dt(dtheta3/dt]

        :param q: state
        :param u: control input
        :return: dq/dt
        """
        # g, m1, l1, m2, l2, m3, l3 = self.params
        # q1, q2, q3, q4, q5, q6 = q

        # TODO: Implement equations of motion

        # return qdot