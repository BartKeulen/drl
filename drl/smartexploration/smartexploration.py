import numpy as np

from drl.replaybuffer import ReplayBufferKD
from drl.smartexploration.policy import LinearGaussianPolicy
from drl.smartexploration.trajectory import Trajectory


class SmartExploration(object):

    def __init__(self,
                 env,
                 T,
                 policy=LinearGaussianPolicy,
                 num_policies=5,
                 num_exec_polices=5,
                 buffer_size=1000000,
                 kernel='gaussian',
                 bandwidth=0.15,
                 leaf_size=100,
                 sample_size=100):
        self.env = env
        self.T = T
        self.policy = policy
        self.num_polices = num_policies
        self.num_exec_policies = num_exec_polices
        self.sample_size = sample_size
        self.buffer = ReplayBufferKD(buffer_size, kernel, bandwidth, leaf_size)

    def get_random_policies(self):
        policies = []
        for i in range(self.num_polices):
            policy = self.policy(self.T, self.env.observation_space.shape[0], self.env.action_space.shape[0])
            policy.init_random()
            policies.append(policy)

        return policies

    def execute_policy(self, policy, obs=None, T=None, render=False):
        T = self.T if T is None else T
        if obs is None:
            obs = self.env.reset()
            self.buffer.new_episode(policy)
        for t in range(T):
            if render:
                self.env.render()

            u = policy.act(t, obs)
            obs_tp1, r, t, _ = self.env.step(u)
            self.buffer.add(obs, u, r, obs_tp1, t)
            obs = obs_tp1
        return obs

    def smart_start(self):
        samples, scores = self.buffer.kd_estimate(self.sample_size)

        self.env.set_initial_states(samples[0])

        argmin_score = np.argmin(scores)

        start_idx = self.buffer.parent_idxes[samples[-1][argmin_score]]
        traj = self.buffer.get_trajectory(start_idx)
        policy = self.buffer.get_policy(start_idx)

        return samples[0][argmin_score], traj, policy


if __name__ == "__main__":
    from drl.env.maze import Maze

    env = Maze.generate_maze(Maze.MEDIUM)

    smart_expl = SmartExploration(env, 1000, num_policies=5)

    st_traj, st_policy, obs = None, None, None
    for i in range(100):
        policies = smart_expl.get_random_policies()

        for policy in policies:
            if st_policy is not None:
                policy = policy.concatenate(st_policy.get(st_traj.T))
            print("Policy length: ", policy.T)
            smart_expl.execute_policy(policy, T=policy.T)

            # traj = smart_expl.buffer.get_trajectory(smart_expl.buffer.parent)
            # env.add_trajectory(traj.get_X(), (255, 0, 0, 255))

        print("Iteration %d, Buffer size: %d" % (i, smart_expl.buffer.size()))
        # rgb_img = smart_expl.buffer.get_rgb_array(env)
        # env.set_density_rgb_array(rgb_img)
        # env.render()


        smart_start, st_traj, st_policy = smart_expl.smart_start()
