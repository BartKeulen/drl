import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, LinearDecay
from drl.utilities import Statistics

# TODO: Use argparse package for running from command line

env_name = "Pendulum-v0"
save_results = False

options_ddpg = {
    'batch_norm': False,
    'l2_critic': 0.01,
    'num_updates_iter': 1
}

options_agent = {
    'render_env': True,
    'num_episodes': 10
}


with tf.Session() as sess:
    env = gym.make(env_name)

    stat = Statistics(sess, env_name, DDPG.get_info(), save=save_results)

    ddpg = DDPG(sess=sess,
                env=env,
                options_in=options_ddpg)

    ddpg.restore('/home/bartkeulen/repositories/drl/drl/../results/test/Pendulum-v0/DDPG/149')

    agent = RLAgent(env=env,
                    algo=ddpg,
                    exploration=None,
                    stat=stat,
                    options_in=options_agent
                    )

    agent.infer()

    sess.close()