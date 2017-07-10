options_algo = {
    'batch_norm': False,
    'l2_critic': 0.,
    'num_updates_iter': 1,
    'hidden_nodes': [400, 300]
}

options_agent = {
    'render_env': False,
    'num_episodes': 1000,
    'max_steps': 200,
    'num_exp': 1,
    'save_freq': 100,
    'record': True
}

options_noise = {
    'mu': 0.,
    'theta': 0.2,
    'sigma': 0.15,
    'start': 500,
    'end': 750
}

