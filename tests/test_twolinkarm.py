from unittest import TestCase
from drl.env.arm import TwoLinkArm
import numpy as np


class TestTwoLinkArm(TestCase):

    def test_pid_controller(self):
        env = TwoLinkArm()
        q = env.reset()

        ths = env.goal
        Ts = 20

        Kp1 = 15
        Kd1 = 7
        Ki1 = 10
        Kp2 = 15
        Kd2 = 10
        Ki2 = 10

        def get_input(q, x_int):
            q1, q2, q3, q4 = q
            u1 = Kp1 * (ths[0] - q1) - Kd1 * q3 + Ki1 * x_int[0]
            u2 = Kp2 * (ths[1] - q2) - Kd2 * q4 + Ki2 * x_int[1]
            return (u1, u2)

        x_int = [0., 0.]
        f_old = np.array([ths[0] - q[0], ths[1] - q[1]])
        f_int = np.array([0., 0.])

        sol = []
        sol.append(q)
        for i in range(int(Ts / env.dt)):
            u = get_input(q, x_int)

            q, _, _, _ = env.step(u)

            env.render()

            f_new = np.array([ths[0] - q[0], ths[1] - q[1]])
            f_int = f_int + (f_old + f_new) * env.dt / 2.
            f_old = f_new
            x_int = [f_int[0], f_int[1]]

            sol.append(q)

        # env.render(close=True)

        sol = np.array(sol)

        import matplotlib.pyplot as plt
        t = np.linspace(0, Ts, int(Ts / env.dt) + 1)

        q1d = np.ones(len(t)) * ths[0]
        q2d = np.ones(len(t)) * ths[1]

        plt.figure()

        plt.subplot(221)
        plt.plot(t, sol[:, 0], 'b', label='Actual')
        plt.plot(t, q1d, 'g--', label='Desired')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel(r'$\theta_1$')
        plt.grid()

        plt.subplot(222)
        plt.plot(t, q1d - sol[:, 0])
        plt.xlabel('t')
        plt.ylabel('error')
        plt.grid()

        plt.subplot(223)
        plt.plot(t, sol[:, 1], 'b', label='Actual')
        plt.plot(t, q2d, 'g--', label='Desired')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel(r'$\theta_2$')
        plt.grid()

        plt.subplot(224)
        plt.plot(t, q2d - sol[:, 1])
        plt.xlabel('t')
        plt.ylabel('error')
        plt.grid()

        plt.show()

        self.assertTrue(True)
