import logging

from .sample import Sample, SampleList
from .trajoptilqg import TrajOptiLQG
from .dynamics import Dynamics


class GuidedExploration(object):

    def __init__(self,
                 num_itr=10,
                 render=False):
        self.num_itr = num_itr
        self.render = render

    def run(self, env, init_traj, cost_func):
        print("Start guided exploration")

        traj_opt = TrajOptiLQG()

        print("Trajopt initialized")

        sample_list = SampleList()
        sample_list.add(init_traj)

        print("Sample list initialized, length: %d", sample_list.len())

        dynamics = Dynamics(env.dynamics)
        dynamics.fit(sample_list.get_X(), sample_list.get_U())

        print("Dynamics initialized")

        traj_distr = traj_opt.backward_pass(sample_list, dynamics, cost_func)

        print("Initial trajectory distribution")

        for itr in range(self.num_itr):
            sample = self.get_sample(env, traj_distr)
            sample_list.add(sample)

            dynamics.fit(sample_list.get_X(), sample_list.get_U())

            traj_distr = traj_opt.optimize(sample_list, traj_distr, dynamics, cost_func)

        return traj_distr

    def get_sample(self, env, traj_distr):
        T, dX, dU = traj_distr.T, traj_distr.dX, traj_distr.dU

        x = env.reset()
        sample = Sample(T, dX, dU)
        for t in range(T):
            if self.render:
                env.render()

            u = traj_distr.act(t, x)
            print("x: %s, u: %s", x, u)
            sample.add(x, u)
            x = env.step(u)

        sample.add(x)
        return sample
