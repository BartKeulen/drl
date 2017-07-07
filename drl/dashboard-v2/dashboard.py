import os
import pickle
from flask import Flask, render_template, request, make_response, redirect


app = Flask(__name__)

DIR = '/'


def get_dir():
    if 'directory' in request.cookies.keys():
        directory = request.cookies['directory']
    else:
        directory = DIR
    return directory


def get_summary_dirs(directory):
    summary_dirs = []
    for root, dirs, files in os.walk(directory):
        if 'summary.p' in files and 'info.p' in files:
            summary_dirs.append(root)

    return summary_dirs


def get_summaries(active_sessions):
    summaries = []
    for i, directory in active_sessions:
        summary = pickle.load(open(os.path.join(directory, 'summary.p'), 'rb'))
        info = pickle.load(open(os.path.join(directory, 'info.p'), 'rb'))
        summaries.append((i, summary, info, directory))
    return summaries


@app.route('/')
def dashboard():
    resp = make_response(render_template('dashboard.html', content='graphs.html', title='Graphs', dir=get_dir()))
    return resp


@app.route('/sessions')
def sessions():
    return render_template('dashboard.html', content='sessions.html', title='Sessions', dir=get_dir(), summary_dirs=get_summary_dirs(get_dir()))


@app.route('/filters')
def filters():
    return render_template('dashboard.html', content='filters.html', title='Filters', dir=get_dir())
