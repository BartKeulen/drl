import time
import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, WhiteNoise

from drl.env import SimpleMaze, MediumMaze

options_ddpg = {
    'batch_norm': False,
    'l2_critic': 0.,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300],
    'prioritized_replay': False
}

options_agent = {
    'render_env': True,
    'num_episodes': 100,
    'max_steps': 2500,
    'num_exp': 1,
}

start = time.time()

env = SimpleMaze()

ddpg = DDPG(env=env,
            options_in=options_ddpg)

noise = OrnSteinUhlenbeckNoise(action_dim=env.action_space.shape[0], scale=env.action_space.high)
# noise = WhiteNoise(env.action_space.shape[0], 1.)

agent = RLAgent(env=env,
                algo=ddpg,
                exploration=noise,
                options_in=options_agent)


with tf.Session() as sess:

    agent.run_experiment(sess)

    sess.close()


end = time.time()