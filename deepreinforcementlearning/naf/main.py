import gym
import tensorflow as tf

from naf import NAF
from deepreinforcementlearning.exploration import *
from deepreinforcementlearning.utils import Statistics

ENV_NAME = "Pendulum-v0"
ALGO_NAME = "NAF"


def main(_):
    with tf.Session() as sess:
        env = gym.make(ENV_NAME)

        stat = Statistics(sess, ENV_NAME, ALGO_NAME, NAF.get_summary_tags())

        noise = OrnSteinUhlenbeckNoise(
            action_dim=env.action_space.shape[0],
            mu=0.,
            theta=0.2,
            sigma=0.15)
        noise_decay = LinearDecay(noise, 50, 100)

        naf = NAF(sess=sess,
                  env=env,
                  stat=stat,
                  learning_rate=0.001,
                  gamma=0.99,
                  tau=0.001,
                  num_updates_iter=5,
                  exploration=noise_decay,
                  buffer_size=1000000,
                  batch_size=64)

        naf.train(num_episodes=200,
                  max_steps=500,
                  render_env=True)

if __name__ == "__main__":
    tf.app.run()