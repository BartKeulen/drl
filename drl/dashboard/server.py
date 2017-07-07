import os
from flask import Flask
from dash import Dash

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key', 'secret')
app = Dash(__name__, server=server, url_base_pathname='/', csrf_protect=False)

app.config.supress_callback_exceptions = True