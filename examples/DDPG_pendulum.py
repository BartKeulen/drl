import tensorflow as tf
from drl.algorithms.ddpg import DDPG
from drl.env import GymEnv
from drl.explorationstrategy import OrnSteinUhlenbeckStrategy
from drl.utilities.scheduler import LinearScheduler

num_experiments = 1

env = GymEnv('Pendulum-v0')

exploration_strategy = OrnSteinUhlenbeckStrategy(action_dim=env.action_space.shape[0])
exploration_decay = LinearScheduler(exploration_strategy, start=100, end=125)

agent = DDPG(env=env,
             l2_critic=0.01,
             batch_norm=False,
             render_env=True,
             num_episodes=100,
             render_freq=10,
             record_freq=13,
             max_steps=200)

with tf.Session() as sess:
    for i in range(num_experiments):
        print("\n\033[1mStarting experiment %d\033[0m" % i)
        agent.train(sess)
