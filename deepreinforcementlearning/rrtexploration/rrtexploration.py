from trajectory import Trajectory
from tree import Tree
import numpy as np


class RRTExploration(object):

    def __init__(self, env, exploration_noise):
        self.env = env
        self.exploration_noise = exploration_noise

        self.tree = Tree()
        self.expand_node = None
        self.expand_state = None

    def explore(self, num_episodes, max_steps, render_env=False):
        self.env.toggle_plot_trajectories()
        cur_trajectory = None
        node_ind = None

        for i_episode in xrange(num_episodes):
            obs = self.env.reset()

            if cur_trajectory is not None:
                split_ind = np.random.randint(1, cur_trajectory.size()-2)

                self.tree.print_tree()

                to_node = self.tree.split_trajectory(node_ind, split_ind)
                states, actions = self.tree.trajectory_to_node(to_node)
                for i in range(actions.shape[0]):
                    next_obs, reward, terminal, info = self.env.step(actions[i])
                obs = next_obs

            i_step = 0
            terminal = False
            self.exploration_noise.reset()

            cur_trajectory = Trajectory()

            while (not terminal) and (i_step < max_steps):
                if render_env:
                    self.env.render()

                action = self.exploration_noise.get_noise()

                cur_trajectory.add_node(obs, action)

                next_obs, reward, terminal, info = self.env.step(action)

                obs = next_obs
                i_step += 1

            self.exploration_noise.increase()

            node, node_ind = self.tree.add_trajectory(cur_trajectory)

            self.env.add_trajectory(cur_trajectory)

            if render_env:
                self.env.render()

            raw_input()


def main():
    import gym_bart
    import gym
    from deepreinforcementlearning.exploration import WhiteNoise, OrnSteinUhlenbeckNoise

    env = gym.make("Double-Integrator-v0")

    noise = WhiteNoise(env.action_space.shape[0], 0., 0.05)
    noise = OrnSteinUhlenbeckNoise(
        action_dim=env.action_space.shape[0],
        mu=0.,
        theta=0.005,
        sigma=0.005)

    agent = RRTExploration(env=env,
                           exploration_noise=noise)

    agent.explore(num_episodes=20,
                  max_steps=100,
                  render_env=True)

if __name__ == "__main__":
    main()