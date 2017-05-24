import gym_bart
import gym
import tensorflow as tf

from ddpg import DDPG
from deepreinforcementlearning.exploration import *
from deepreinforcementlearning.utils import Statistics

# ENV_NAME = "Pendulum-v0"
# ENV_NAME = "MountainCarContinuous-v0"
ENV_NAME = "Double-Integrator-v0"
ALGO_NAME = "DDPG"


def main(_):
    with tf.Session() as sess:
        env = gym.make(ENV_NAME)

        stat = Statistics(sess, ENV_NAME, ALGO_NAME, DDPG.get_summary_tags())

        noise = WhiteNoise(env.action_space.shape[0], 0., 0.05)
        # noise = OrnSteinUhlenbeckNoise(
        #     action_dim=env.action_space.shape[0],
        #     mu=0.,
        #     theta=0.005,
        #     sigma=0.005)
        # noise_decay = LinearDecay(noise, 3, 5)
        # noise = ConstantNoise(env.action_space.shape[0], 0.)

        ddpg = DDPG(sess=sess,
                    env=env,
                    stat=stat,
                    learning_rate_actor=0.0001,
                    learning_rate_critic=0.001,
                    gamma=0.99,
                    tau=0.001,
                    hidden_nodes=[20, 20],
                    exploration=noise,
                    buffer_size=10000,
                    batch_size=64)

        ddpg.train(num_episodes=2,
                   max_steps=100,
                   render_env=True)

if __name__ == "__main__":
    tf.app.run()