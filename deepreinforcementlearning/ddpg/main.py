import gym_bart
import gym
import tensorflow as tf

from ddpg import DDPG
from deepreinforcementlearning.exploration import *
from deepreinforcementlearning.utils import Statistics

ENV_NAME = "Pendulum-v0"
# ENV_NAME = "InvertedDoublePendulum-v1"
# ENV_NAME = "MountainCarContinuous-v0"
# ENV_NAME = "Double-Integrator-v0"
ALGO_NAME = "DDPG"
SAVE = False
NUM_EXP = 1

SETTINGS = {
    'learning_rate_actor': 0.0001,
    'learning_rate_critic': 0.001,
    'gamma': 0.99,
    'tau': 0.001,
    'hidden_nodes': [400, 300],
    'batch_norm': True,
    'batch_size': 64,
    'buffer_size': 1000000,
    'num_updates_iter': 5
}


def main(_):
    for _ in range(NUM_EXP):
        # TODO: make it so multiple sessions can run
        with tf.Session() as sess:
            env = gym.make(ENV_NAME)

            stat = Statistics(sess, ENV_NAME, ALGO_NAME, DDPG.get_summary_tags(), settings=SETTINGS, save=SAVE)

            # noise = WhiteNoise(env.action_space.shape[0], 0., 0.05)
            noise = OrnSteinUhlenbeckNoise(
                action_dim=env.action_space.shape[0],
                mu=0.,
                theta=0.005,
                sigma=0.005)
            noise_decay = LinearDecay(noise, 100, 125)
            # noise = ConstantNoise(env.action_space.shape[0], 0.)

            ddpg = DDPG(sess=sess,
                        env=env,
                        stat=stat,
                        exploration=noise_decay,
                        **SETTINGS)

            ddpg.train(num_episodes=21,
                       max_steps=200,
                       render_env=False)

            sess.close()


if __name__ == "__main__":
    tf.app.run()