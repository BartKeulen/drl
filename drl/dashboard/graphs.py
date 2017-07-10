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
            id='graphs'
        )
    ]
)


# Update callback for graphs
@app.callback(Output('graphs', 'children'),
              [Input('ma-slider', 'value')],
              events=[Event('interval-component', 'interval')])
def update_graphs(ma_param):
    summaries = get_summaries(app.states['ACTIVE_SESSIONS'])
    tags = get_tags(summaries)
    graphs = [html.Div(
                className='col-6',
                children=[
                    dcc.Graph(
                        id=tag,
                        figure=get_graph(summaries, tag, ma_param)
                    )
                ]
            ) for tag in tags]
    return graphs