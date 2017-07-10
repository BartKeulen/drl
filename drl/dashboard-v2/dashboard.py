import os
import pickle
import datetime
from flask import *
from functools import wraps, update_wrapper

app = Flask(__name__)


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


def get_dir():
    if 'directory' in request.cookies.keys():
        directory = request.cookies['directory']
    else:
        directory = '/'
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


@app.route('/', methods=['GET'])
def get_dashboard():
    return render_template('dashboard.html', dir=get_dir())


@app.route('/summaries', methods=['GET'])
def get_summaries():
    summary_dirs = get_summary_dirs(request.cookies['directory'])
    active_sessions = request.cookies['active_sessions']
    sessions = {}
    for summary_dir in summary_dirs:
        session_split = summary_dir.split('/')
        env = session_split[-7]
        algo = session_split[-6]
        time_stamp = datetime.datetime(int(session_split[-5]), int(session_split[-4]), int(session_split[-3]),
                                       int(session_split[-2][:2]), int(session_split[-2][2:]))
        run = session_split[-1].split('_')[-1]
        if summary_dir in active_sessions:
            active = True
        else:
            active = False

        if env not in sessions.keys():
            sessions[env] = {}
        if algo not in sessions[env]:
            sessions[env][algo] = [{
                'date': time_stamp,
                'runs': [{'id': run, 'path': summary_dir, 'active': active}],
            }]
        else:
            exists = False
            for sess in sessions[env][algo]:
                if sess['date'] == time_stamp:
                    sess['runs'].append({'id': run, 'path': summary_dir, 'active': active})
                    exists = True
            if not exists:
                sessions[env][algo].append({
                    'date': time_stamp,
                    'runs': [{'id': run, 'path': summary_dir, 'active': active}],
                })

    return jsonify({'summaries': sessions})


@app.route('/active_sessions', methods=['POST'])
def get_active_sessions():
    active_sessions = request.get_json()
    summaries = []
    for path in active_sessions:
        summary = pickle.load(open(os.path.join(path, 'summary.p'), 'rb'))
        info = pickle.load(open(os.path.join(path, 'info.p'), 'rb'))
        summaries.append((path, summary, info))

    return jsonify(summaries)