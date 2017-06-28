import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, LinearDecay
from drl.utilities import StatisticsTF

# TODO: Use argparse package for running from command line

env_name = "HalfCheetah-v1"

options_ddpg = {
    'batch_norm': False,
    'l2_critic': 0.,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300]
}

options_agent = {
    'render_env': False,
    'num_episodes': 7500,
    'max_steps': 1000,
    'num_exp': 5
}


with tf.Session() as sess:
    env = gym.make(env_name)

    ddpg = DDPG(sess=sess,
                env=env,
                options_in=options_ddpg)

    ddpg.restore_model('/home/bartkeulen/repositories/drl/drl/../results/tmp/HalfCheetah-v1/DDPG/2017/06/27/1840/run_0')

    agent = RLAgent(env=env,
                    algo=ddpg,
                    exploration=None,
                    stat=None,
                    options_in=options_agent
                    )

    agent.test(10, 1000, True)

    sess.close()