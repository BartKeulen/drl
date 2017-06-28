import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, LinearDecay
from drl.env import Pendulum

# TODO: Use argparse package for running from command line

env_name = 'HalfCheetah-v1'
save_results = True

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
    'num_exp': 5,
    'save_freq': 250,
    'record': True
}


with tf.Session() as sess:
    env = gym.make('HalfCheetah-v1')

    ddpg = DDPG(sess=sess,
                env=env,
                options_in=options_ddpg)

    noise = OrnSteinUhlenbeckNoise(
        action_dim=env.action_space.shape[0],
        mu=0.,
        theta=0.2,
        sigma=0.15)
    noise = LinearDecay(noise, 300, 500)

    agent = RLAgent(env=env,
                    algo=ddpg,
                    exploration=noise,
                    options_in=options_agent
                    )

    agent.run_experiment()

    sess.close()
