import os, pickle
import numpy as np

import plotly.graph_objs as go


def get_directories(path):
    directories = []
    for root, dirs, files in os.walk(path):
        if 'summary.p' in files and 'info.p' in files:
            directories.append(root)

    return directories


def get_label(i, dir):
    summary = pickle.load(open(os.path.join(dir, 'info.p'), 'br'))
    label = str(i) + ' - ' + summary['info']['name']
    if 'run' in summary['info']:
        label += ' - run ' + str(summary['info']['run'])
    if 'timestamp' in summary['info']:
        label += ' - ' + summary['info']['timestamp']
    return label


def get_summaries(active_sessions):
    summaries = []
    for i, directory in active_sessions:
        summary = pickle.load(open(os.path.join(directory, 'summary.p'), 'rb'))
        info = pickle.load(open(os.path.join(directory, 'info.p'), 'rb'))
        summaries.append((i, summary, info, directory))
    return summaries


def get_tags(summaries):
    tags = []
    for summary in summaries:
        for tag in summary[1]['values']:
            if tag not in tags:
                tags.append(tag)
    return tags


# Returns single graph
def get_graph(summaries, tag, ma_param):
    data = []
    for summary in summaries:
        value = moving_average(summary[1]['values'][tag], ma_param)
        trace = go.Scatter(
            x=summary[1]['episode'][:-1],
            y=value,
            mode='lines',
            line=dict(
                width=1
            ),
            name=str(summary[0])
        )
        data.append(trace)
    layout = dict(title=tag)
    fig = dict(data=data, layout=layout)
    return fig


def mean_sessions_filter(values):
    values = np.array(values)
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    mean = np.mean(values, axis=0)

    return [mean, upper, lower]


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n