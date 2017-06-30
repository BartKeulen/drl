import threading

import gym
import tensorflow as tf

from drl.rlagent import RLAgent
from drl.ddpg import DDPG
from drl.exploration import OrnSteinUhlenbeckNoise, LinearDecay

env_name = 'Pendulum-v0'
save_results = False

options_ddpg = {
    'batch_norm': False,
    'l2_critic': 0.,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300]
}

options_agent = {
    'render_env': False,
    'num_episodes': 10,
    'max_steps': 200,
    'num_exp': 1,
    'print': False
}

options_noise = {
    'start': 100,
    'end': 125
}

def thread_func(sess, thread_id):
    env = gym.make(env_name)

    ddpg = DDPG(env=env,
                options_in=options_ddpg)

    noise = OrnSteinUhlenbeckNoise(action_dim=env.action_space.shape[0])
    noise = LinearDecay(noise, options_in=options_noise)

    agent = RLAgent(env=env,
                    algo=ddpg,
                    exploration=noise,
                    options_in=options_agent,
                    save=save_results)

    agent.run_single_experiment(thread_id)

    print('Thread %d finished' % thread_id)

with tf.Session() as sess:
    threads = [threading.Thread(target=thread_func
                                , args=(sess, i), name='Thread-%d' % i) for i in range(5)]

    sess.run(tf.global_variables_initializer())

    try:
        coord = tf.train.Coordinator()

        for thread in threads:
            thread.start()

        coord.join(threads, stop_grace_period_secs=10)

    except RuntimeError:
        raise RuntimeError('One of the threads took more than 10s to stop after request.stop() was called.')
    except Exception as ex:
        raise Exception('Exception that was passed to coord.request_stop(): ', str(ex))

    sess.close()