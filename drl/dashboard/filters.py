import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html

from drl.dashboard.server import app
from drl.dashboard.methods import *

layout = html.Div(
    className='col-sm-12',
    children=[
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='col-sm-12',
                    children=[
                        html.H5('Choose sessions'),
                        dcc.Checklist(
                            id='sessions_filter',
                            inputClassName='form-check-input',
                            options=[],
                            values=[],
                            labelStyle={'display': 'inline-block'},
                            labelClassName='form-check-label'
                        )
                    ]
                )
            ]
        ),
        html.Hr(),
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='col-sm-12',
                    children=[
                        html.H5('Apply filter'),
                        dcc.RadioItems(
                            options=[
                                {'label': i, 'value': i} for i in ['Average episodes']
                            ],
                            id='filters',
                            labelClassName='btn btn-primary',
                            inputStyle={
                                'visibility': 'hidden'
                            },
                            labelStyle={
                                'cursor': 'pointer'
                            }
                        )
                    ]
                )
            ]
        ),
        html.Hr(),
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='col-sm-12',
                    children=[
                        html.H5('Active filters')
                    ]
                )
            ]
        ),
        html.Div(
            id='null2'
        )
    ]
)


@app.callback(Output('sessions_filter', 'options'),
              [Input('path', 'value')],
              events=[Event('interval-component', 'interval')])
def update_sessions(path):
    directories = get_directories(path)
    return [{'label': get_label(i, dir), 'value': (i, dir)}
            for i, dir in zip(range(len(directories)), directories)]


@app.callback(Output('null2', 'children'),
              [Input('session_filters', 'values')])
def applay_average_filter(sessions):
    print(sessions)
    return []