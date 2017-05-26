import os


def get_summary_dir(dir_name, env_name, algo_name, settings=None, save=False):
    if save:
        tmp = 'eval'
    else:
        tmp = 'test'

    summary_dir = os.path.join(dir_name, tmp, env_name, algo_name)

    if settings is None:
        summary_dir = os.path.join(summary_dir, 'other')
    else:
        for key, value in settings.items():
            summary_dir = os.path.join(summary_dir, '%s=%s' % (key, value))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    count = 0
    for f in os.listdir(summary_dir):
        child = os.path.join(summary_dir, f)
        if os.path.isdir(child):
            count += 1

    return os.path.join(summary_dir, str(count))
