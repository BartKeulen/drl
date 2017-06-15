# import gym_bart
import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, LinearDecay
from drl.replaybuffer import ReplayBuffer
from drl.utilities import Statistics

# TODO: Use argparse package for running from command line

ENV_NAME = "Pendulum-v0"
SAVE = False
NUM_EXP = 1

options_ddpg = {
    'batch_norm': False,
    'l2': 0.,
    'num_updates_iter': 5
}

options_agent = {
    'render_env': False
}


def main(_):
    with tf.Session() as sess:
        env = gym.make(ENV_NAME)

        stat = Statistics(sess, ENV_NAME, DDPG.get_info(), save=SAVE)

        ddpg = DDPG(sess=sess,
                    env=env,
                    options_in=options_ddpg)

        noise = OrnSteinUhlenbeckNoise(
            action_dim=env.action_space.shape[0],
            mu=0.,
            theta=0.2,
            sigma=0.15)
        noise = LinearDecay(noise, 100, 125)

        replay_buffer = ReplayBuffer(DDPG.get_info()[1]['buffer_size'])

        agent = RLAgent(env=env,
                        algo=ddpg,
                        exploration=noise,
                        replay_buffer=replay_buffer,
                        stat=stat,
                        options_in=options_agent
                        )

        agent.run_experiment()

        sess.close()

if __name__ == "__main__":
    tf.app.run()
