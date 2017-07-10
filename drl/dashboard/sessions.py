import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html

from drl.dashboard.server import app
from drl.dashboard.methods import *

layout = html.Div(
    className='col-sm-12',
    children=[
        dcc.Checklist(
            id='sessions',
            inputClassName='form-check-input',
            options=[],
            values=[],
            labelStyle={'display': 'inline-block'},
            labelClassName='form-check-label'
        )
    ],
)

@app.callback(Output('sessions', 'values'),
              events=[Event('interval-component', 'interval')])
def update_sessions_values():
    return app.states['ACTIVE_SESSIONS']

@app.callback(Output('sessions', 'options'),
              [Input('path', 'value')],
              events=[Event('interval-component', 'interval')])
def update_sessions(path):
    directories = get_directories(path)
    return [{'label': get_label(i, dir), 'value': (i, dir)}
            for i, dir in zip(range(len(directories)), directories)]

@app.callback(Output('null', 'children'),
              [Input('sessions', 'values')])
def update_active_sessions(sessions):
    if len(sessions) != 0:
        app.states['ACTIVE_SESSIONS'] = sessions
    return []