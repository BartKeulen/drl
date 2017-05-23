import os


def get_summary_dir(dir, env, algo):
    summary_dir = os.path.join(dir, env, algo)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    count = 0
    for f in os.listdir(summary_dir):
        child = os.path.join(summary_dir, f)
        if os.path.isdir(child):
            count += 1

    return os.path.join(summary_dir, str(count))