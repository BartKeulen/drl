import tensorflow as tf
import numpy as np
from tqdm import tqdm

from drl.algorithms.ddpg import DDPG
from drl.env.environment import GymEnv, save_recording
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy

run = 1
path = '/home/bartkeulen/results/HalfCheetah-v1/DDPG/%d/final' % run

env = GymEnv('HalfCheetah-v1')

agent = DDPG(env)

exploration_strategy = OrnSteinUhlenbeckStrategy(env.action_space.shape[0])

with tf.Session() as sess:
    agent.set_session(sess)
    agent.restore_model(sess, path)

    obs = env.reset()
    for i in tqdm(range(500)):
        env.render(record=True)
        a = agent.get_action(obs)
        obs, _, _, _ = env.step(a[0])

    input()
    save_recording('/home/bartkeulen/results/', '%d' % run)