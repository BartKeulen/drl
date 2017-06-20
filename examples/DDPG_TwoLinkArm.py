import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, LinearDecay
from drl.utilities import Statistics
from drl.env.arm import TwoLinkArm

# TODO: Use argparse package for running from command line

env_name = "TwoLinkArm"
save_results = False

options_ddpg = {
    'lr_actor': 0.000001,
    'lr_critic': 0.00001,
    'batch_norm': True,
    'l2_critic': 0.01,
    'l2_actor': 0.01,
    'num_updates_iter': 5
}

options_agent = {
    'render_env': True,
    'render_freq': 1,
    'num_episodes': 5000,
    'max_steps': 100
}


with tf.Session() as sess:
    env = TwoLinkArm(g=0.)

    stat = Statistics(sess, env_name, DDPG.get_info(), save=save_results)

    ddpg = DDPG(sess=sess,
                env=env,
                options_in=options_ddpg)

    noise = OrnSteinUhlenbeckNoise(
        action_dim=env.action_space.shape[0],
        mu=0.,
        theta=0.2,
        sigma=0.15)
    noise = LinearDecay(noise, 100, 125)

    agent = RLAgent(env=env,
                    algo=ddpg,
                    exploration=noise,
                    stat=stat,
                    options_in=options_agent
                    )

    agent.run_experiment()

    sess.close()
