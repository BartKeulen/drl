import os
import pickle
import argparse

import numpy as np

import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go


def get_directories():
    directories = []
    for root, dirs, files in os.walk(path):
        if 'summary.p' in files and 'info.p' in files:
            directories.append(root)

    if len(directories) == 0:
        print('No summaries where find in this directory tree.')
        exit(0)

    return directories

def get_label(i, dir):
    summary = pickle.load(open(os.path.join(dir, 'info.p'), 'br'))
    label = str(i) + ' - ' + summary['info']['name'] + ' - run: ' + str(summary['info']['run']) + ' - timestamp: ' + \
            summary['info']['timestamp']
    return label

ma_param = 5
active_sessions = []

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, required=True)
parser.add_argument('-interval', type=int)
args = parser.parse_args()
path = args.path
if args.interval is not None:
    update_interval = args.interval
else:
    update_interval = 1  # in seconds

directories = get_directories()
print(directories)

# Initialize app
app = dash.Dash()

# Load css and js
app.css.append_css({
    "external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css"
})
app.css.append_css({
    "external_url": "https://v4-alpha.getbootstrap.com/examples/dashboard/dashboard.css"
})
app.scripts.append_script({
    "external_url": "https://code.jquery.com/jquery-3.1.1.slim.min.js"
})
app.scripts.append_script({
    "external_url": "https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js"
})
app.scripts.append_script({
    "external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js"
})

# Main dahsboard layout
app.layout = html.Div(children=[
    html.Nav(className='navbar navbar-inverse fixed-top bg-inverse', children=[
        html.A(className='navbar-brand', href='#', children='Dashboard')
    ]),
    html.Div(className='container-fluid row', children=[
        html.Div(className='col-sm-3 col-md-2 hidden-xs-down bg-faded sidebar', id='sidebar', children=[
            html.Div(className='container', id='general-settings', children=[
                html.H5('General settings'),
                html.Span('Moving average filter'),
                html.Div([
                    dcc.Slider(
                        id='ma-slider',
                        min=1,
                        max=10,
                        value=5,
                        step=1,
                        marks={str(i): str(i) for i in range(1, 11)}
                    ),
                ], style={'margin-bottom': '30px'})
            ]),
            html.Div(className='container form-check', children=[
                html.H5('Sessions'),
                dcc.Checklist(
                    id='sessions',
                    inputClassName='form-check-input',
                    options=[{'label': get_label(i, dir), 'value': (i, dir)}
                             for i, dir in zip(range(len(directories)), directories)],
                    values=[],
                    labelStyle={'display': 'inline-block'},
                    labelClassName='form-check-label'
                )
            ]),
            # html.Div(className='container', children=[
            #     html.H5('Hyper parameters'),
            #     # html.Table(className='table table-sm', children=[
            #     #     html.Tr([
            #     #     html.Td(children=tag),
            #     #     html.Td(children=str(info['algo'][tag]))
            #     # ]) for tag in info['algo']]),
            # ])
        ]),
        html.Div(className='col-sm-9 offset-sm-3 col-md-10 offset-md-2 pt-3', id='main', children=[
            # html.H1(info['info']['env']),
            html.Div(className='row', id='graphs')
        ])
    ]),
    dcc.Interval(
        id='interval-component',
        interval=update_interval*1000
    ),
    html.Div(
        id='null'
    )
])

@app.callback(Output('null', 'children'),
              [Input('sessions', 'values'), Input('ma-slider', 'value')])
def update_settings(sessions, new_ma_param):
    global ma_param, active_sessions
    ma_param = new_ma_param
    active_sessions = sessions


# Update callback for graphs
@app.callback(Output('graphs', 'children'),
              events=[Event('interval-component', 'interval')])
def update_graphs():
    global directories
    directories = get_directories()
    summaries = get_summaries()
    tags = get_tags(summaries)
    graphs = [html.Div(
                className='col-6',
                children=[
                    dcc.Graph(
                        id=tag,
                        figure=get_graph(summaries, tag)
                    )
                ]
            ) for tag in tags]
    return graphs


def get_summaries():
    global directories, active_sessions
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
def get_graph(summaries, tag):
    data = []
    for summary in summaries:
        # info = summary[1]['info']
        # name = info['name'] + ' - ' + info['timestamp'] + ' - ' + str(info['run'])
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


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    app.run_server(debug=True)