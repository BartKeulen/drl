from importlib.machinery import SourceFileLoader
import argparse
from argparse import RawTextHelpFormatter

import gym
import tensorflow as tf
from drl.ddpg import DDPG
from drl.env import *
from drl.exploration import *
from drl.rlagent import RLAgent

# Dicts holding the available environments, algorithms, noise type and noise decay types
envs = {'pendulum': Pendulum, 'twolinkarm': TwoLinkArm}
algos = {'ddpg': DDPG}
noises = {'ornsteinuhlenbeck': OrnSteinUhlenbeckNoise, 'white': WhiteNoise, 'constant': ConstantNoise}
noise_decays = {'linear': LinearDecay, 'exponential': ExponentialDecay}

# Initialize argument parser
parser = argparse.ArgumentParser(description="""
This file is used to easily run experiments from the command line.

The environment, algorithm, noise type and noise decay type can be given through command line options. More detailed options are passed using a configurion.py file. Default settings are used when no configuration is given, the settings can be found in the docstrings of the source files. 

An example of a configuration file looks like:

    options_algo = {
        'batch_norm': False,
        'l2_critic': 0.,
        'num_updates_iter': 1,
        'hidden_nodes': [400, 300]
    }
    
    options_agent = {
        'render_env': False,
        'num_episodes': 7500,
        'max_steps': 1000,
    }
    
    options_noise = {
        'mu': 0.,
        'theta': 0.2,
        'sigma': 0.15,
        'start': 1000,
        'end': 2000
    }

Not all options need to be filled. The options_algo part describes the algorithm settings, the options_agent the learning agent settings and hte options_noise the noise and noise decay settings.
""", formatter_class=RawTextHelpFormatter)

parser.add_argument('-env', type=str, required=True, help='Environment, can be gym environment or one of the '
                                                          'environments in this package.')
parser.add_argument('--gym', action='store_true', help='Add tag when OpenAI gym environment is used.')
parser.add_argument('-algo', type=str, default='ddpg', choices=list(algos.keys()), help='Algorithm')
parser.add_argument('-noise', type=str, choices=list(noises.keys()),
                    help='Exploration noise to be used')
parser.add_argument('-noise_decay', type=str, choices=list(noise_decays.keys()), help='Adds noise decay')

parser.add_argument('-path', type=str, help='Path to configuration file.')

parser.add_argument('--save', action='store_true', help='Save the results in eval folder else in tmp folder.')

args = parser.parse_args()

# Check if options chosen are correct
if not args.gym and args.env not in envs:
    raise Exception(args.env, ' is not a valid environment, if you are using a gym environment add --gym otherwise choose from the available choices: ', list(envs.keys()))
if args.algo not in algos:
    raise Exception(args.algo, ' is not a valid algorithm, choose from the available choices: ', list(algos.keys()))
if args.noise and args.noise not in noises:
    raise Exception(args.noise, ' is not a valid noise type, choose from the available choices: ', list(noises.keys()))
if args.noise_decay and args.noise_decay not in noise_decays:
    raise Exception(args.noise_decay, ' is not a valid noise decay type, choose from the available choices: ', list(noise_decays.keys()))

# Load configuration file
if args.path:
    config = SourceFileLoader('module.name', args.path).load_module()
    options_algo = config.options_algo
    options_agent = config.options_agent
    options_noise = config.options_noise
else:
    options_algo = {}
    options_agent = {}
    options_noise = {}

# Run
with tf.Session() as sess:
    if args.gym:
        env = gym.make(args.env)
        options_agent['max_steps'] = env._max_episode_steps
    else:
        env = envs[args.env]

    algo = algos[args.algo](sess=sess,
                            env=env,
                            options_in=options_algo)

    if args.noise:
        noise = noises[args.noise](env.action_space.shape[0],
                                   options_in=options_noise)
        if args.noise_decay:
            noise = noise_decays[args.noise_decay](noise,
                                                   options_in=options_noise)
    else:
        noise = None

    agent = RLAgent(env=env,
                    algo=algo,
                    exploration=noise,
                    options_in=options_agent,
                    save=args.save)

    agent.run_experiment()

    sess.close()
