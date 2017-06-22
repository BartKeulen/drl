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
    'batch_norm': True,
    'l2_critic': 0.01,
    'num_updates_iter': 5
}

options_agent = {
    'render_env': False,
    'num_episodes': 200,
    'save_freq': 5,
    'record': True
}


with tf.Session() as sess:
    env = gym.make(env_name)

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
