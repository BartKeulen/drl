import argparse
from collections import OrderedDict

import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html


from drl.dashboard.server import app
import drl.dashboard.graphs as graphs
import drl.dashboard.sessions as sessions
import drl.dashboard.filters as filters

app.states['ACTIVE_SESSIONS'] = []
app.states['filters'] = []

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

# Load css and js
app.css.append_css({
    "external_url": (
        "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css",
        "https://v4-alpha.getbootstrap.com/examples/dashboard/dashboard.css"
    )
})
app.scripts.append_script({
    "external_url": (
        "https://code.jquery.com/jquery-3.1.1.slim.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js",
        "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js"
    )
})

chapters = {
    'graphs': graphs.layout,
    'sessions': sessions.layout,
    'filters': filters.layout
}

# Navigation
navbar = html.Nav(
    className='navbar navbar-toggleable-md navbar-inverse fixed-top bg-inverse',
    children=[
        html.A(
            className='navbar-brand',
            href='#',
            children='Dashboard'
        ),
        html.Div(
            className='collapse navbar-collapse',
            id='navbarsExampleDefault',
            children=[
                html.Ul(
                    className='navbar-nav mr-auto',
                    children=[
                        dcc.RadioItems(
                            options=[
                                {'label': i, 'value': i} for i in chapters.keys()
                            ],
                            value='graphs',
                            id='toc',
                            labelClassName='nav-item',
                            inputStyle={
                                'visibility': 'hidden'
                            },
                            labelStyle={
                                'color': 'white',
                                'cursor': 'pointer'
                            }
                        )
                    ]
                ),
                html.Form(
                    className='form-inline mt-8 mt-md-0',
                    children=[
                        dcc.Input(
                            className='form-control mr-sm-2',
                            id='path',
                            value=path
                        )
                    ]
                )
            ]
        )
    ]
)

# Left sidebar
sidebar = html.Div(
    className='col-sm-3 col-md-2 hidden-xs-down bg-faded sidebar',
    id='sidebar',
    children=[
        html.Div(
            className='container',
            id='general-settings',
            children=[
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
            ]
        ),
        html.Div(
            className='container form-check',
            children=[
                html.H5('Active sessions'),
                html.Ul(
                    id='active-sessions',
                    children=[]
                )
            ]
        ),
    ]
)

# Main content area
content = html.Div(
    className='col-sm-9 offset-sm-3 col-md-10 offset-md-2 pt-3',
    id='main',
    children=[]
)

# Layout
app.layout = html.Div([
    html.Meta(name='viewport', content='width=device-width, initial-scale=1.0'),
    html.Meta(
        name='description',
        content=('Dashboard for viewing the (live) results of your reinforcement learning session.')
    ),
    navbar,
    html.Div(className='container-fluid row', children=[
        sidebar,
        content
    ]),
    dcc.Interval(
        id='interval-component',
        interval=update_interval*1000
    ),
    html.Div(
        id='null',
        children=[]
    )
])

app.title = 'DRL Dashboard'


@app.callback(Output('main', 'children'), [Input('toc', 'value')])
def display_content(selected_chapter):
    content = [
        html.H1(selected_chapter),
        html.Div(
            className='row',
            children=[chapters[selected_chapter]]
        )
    ]
    return content


@app.callback(Output('active-sessions', 'children'),
              events=[Event('interval-component', 'interval')])
def display_active_sessions():
    return [html.Li(children=session) for session in app.states['ACTIVE_SESSIONS']]


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, port=8050)