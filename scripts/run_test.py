import os
import pickle
import argparse

import gym
import tensorflow as tf
from drl.ddpg import DDPG
from drl.env import *
from drl.rlagent import RLAgent

envs = {'pendulum': Pendulum, 'twolinkarm': TwoLinkArm}
algos = {'DDPG': DDPG}

parser = argparse.ArgumentParser(description="""
This file is used for testing learned models from the command line.

The path must contain an info.p file containing all options for the learning session. The path should also contain Tensorflow model.* files containing the state of the model.
""")

parser.add_argument('-path', required=True, help='Path to folder holding the session information. (folder containing info.p file)')
parser.add_argument('-episode', required=False, help='Use a specific episode saving point to load')
parser.add_argument('--gym', action='store_true', help='Add tag when OpenAI gym environment is used.')
parser.add_argument('-num_episodes', type=int, default=10, help='Number of episodes to execute')
parser.add_argument('-max_steps', type=int, default=1000, help='Maximum number of steps per episode')
parser.add_argument('--render_env', action='store_true', help='Render the environment')
parser.add_argument('--record', action='store_true', help='Record the render in .mp4 file')

args = parser.parse_args()

path_info = os.path.join(args.path, 'info.p')
path_model = args.path
if args.episode:
    path_model = os.path.join(path_model, args.episode)
path_model = os.path.join(path_model, 'model')

if not os.path.isfile(path_info):
    raise Exception('No info.p file was found on: ' + path_info)
if not os.path.isfile(path_model + '.index'):
    raise Exception('No model was found on: ' + path_model)

config = pickle.load(open(path_info, 'rb'))

env_name = config['info']['env']
algo_name = config['info']['name']
with tf.Session() as sess:
    if args.gym:
        env = gym.make(env_name)
    else:
        env = envs[env_name]

    algo = algos[algo_name](sess=sess,
                            env=env,
                            options_in=config['algo'])

    algo.restore_model(path_model)

    agent = RLAgent(env=env,
                    algo=algo,
                    exploration=None,
                    options_in=config['agent'])

    agent.test(args.num_episodes, args.max_steps, args.render_env, args.record)

    sess.close()