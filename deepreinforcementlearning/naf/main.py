import gym
import tensorflow as tf

from naf import NAF
from deepreinforcementlearning.exploration import *
from deepreinforcementlearning.utils import Statistics

# TODO: Make examples folder for scripts that work
# TODO: Use argparse package for running from command line

ENV_NAME = "Pendulum-v0"
ALGO_NAME = "NAF"
SAVE = False
NUM_EXP = 1

SETTINGS = {
    'learning_rate': 0.0001,
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

            stat = Statistics(sess, ENV_NAME, ALGO_NAME, NAF.get_summary_tags(), settings=SETTINGS)

            noise = OrnSteinUhlenbeckNoise(
                action_dim=env.action_space.shape[0],
                mu=0.,
                theta=0.15,
                sigma=0.3)
            noise_decay = LinearDecay(noise, 100, 125)

            naf = NAF(sess=sess,
                      env=env,
                      stat=stat,
                      exploration=noise_decay,
                      **SETTINGS)

            naf.train(num_episodes=500,
                      max_steps=200,
                      render_env=True)

if __name__ == "__main__":
    tf.app.run()
