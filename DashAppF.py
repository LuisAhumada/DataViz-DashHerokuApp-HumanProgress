#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import dash
from dash import no_update
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
#
import re
import base64
import io
from io import BytesIO
from keras.models import load_model
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from random import randrange
import pickle
import copy
import pathlib
import urllib.request
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dash
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
from plotly.subplots import make_subplots

# In[ ]:


df = pd.read_csv('/Users/luisabrigo/HumanProgress/Viz/df_all.csv')

# In[ ]:


df_indicators = list(df.columns.drop(['Unnamed: 0', "Countries", 'Year']))

# In[ ]:


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview"
)

app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Human Progress Data",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Modeling Dashboard", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),

                html.Div(
                    [
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="https://humanprogress.org",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Select a year range:",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=1960,
                            max=2017,
                            value=[1990, 2010],
                            className="dcc_control",
                        ),
                        html.P("Select Indicators:", className="control_label"),
                        dcc.RadioItems(
                            id="well_status_selector",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="well_statuses",
                            options=[{'label': i, 'value': i} for i in df_indicators],
                            multi=True,
                            # value=list(INDICATORS.keys()),
                            className="dcc_control",
                        ),
                        html.P("Select Countries:", className="control_label"),
                        dcc.RadioItems(
                            id="country_selector",
                            value="Chile",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="country_types",
                            options=[{'label': j, 'value': j} for j in df['Countries'].unique()],
                            multi=True,
                            # value=list(COUNTRIES.values()),
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                )

            ],
            className="row flex-display",
        ),
        html.Div(
            [
                dcc.Graph(id='graph1',
                          style={'height': '450px', 'width': '50%', 'display': 'inline-block'}
                          ),
                dcc.Graph(id='graph2',
                          style={'height': '450px', 'width': '50%', 'display': 'inline-block'}
                          )
            ]
        ),

        html.Div(
            [
                dcc.Graph(id='graph3',
                          style={'height': '450px', 'width': '50%', 'display': 'inline-block'}
                          ),
                dcc.Graph(id='graph4',
                          style={'height': '450px', 'width': '50%', 'display': 'inline-block'}
                          )
            ]
        )

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"}
)


@app.callback(
    Output('graph1', 'figure'),
    [Input('year_slider', 'value')],
    [Input('well_statuses', 'value')],
    [Input('country_types', 'value')
     ])
def update_figure(value, value1, value2):
    dff = df.loc[:, ['Countries', 'Year', value1[0], value1[1]]]
    dfff = dff.loc[dff['Countries'] == value2[0]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]
    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dffff['Year'], y=dffff[value1[0]], name= str(value2[0][:5]) + " - " + str(value1[0][:10]),
                         line=dict(color='royalblue', width=4)), secondary_y=False)
    fig.add_trace(go.Scatter(x=dffff['Year'], y=dffff[value1[1]], name= str(value2[0][:5]) + " - " + str(value1[1][:10]),
                         line=dict(color='royalblue', width=4, dash='dot')), secondary_y=True)

    fig.update_yaxes(title_text=value1[0][:15], secondary_y=False)
    fig.update_yaxes(title_text=value1[1][:15], secondary_y=True)
    fig.update_xaxes(title_text="Year")
    fig.update_layout(
        title_text="Double Y Axis",
        font_family="Roboto",
        font_color="grey",
        title_font_family="Roboto",
        title_font_color="grey",
    )

    return fig


@app.callback(Output('graph2', 'figure'),
              [Input('year_slider', 'value')],
              [Input('well_statuses', 'value')],
              [Input('country_types', 'value')
               ])
def update_figure(value, value1, value2):
    dff = df.loc[:, ['Countries', 'Year', value1[2]]]
    dfff = dff.loc[dff['Countries'] == value2[1]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]
    return go.Figure(data=go.Scatter(x=dffff['Year'], y=dffff[value1[2]]))


@app.callback(Output('graph3', 'figure'),
              [Input('year_slider', 'value')],
              [Input('well_statuses', 'value')],
              [Input('country_types', 'value')])
def update_figure(value, value1, value2):
    dff = df.loc[:, ['Countries', 'Year', value1[3]]]
    dfff = dff.loc[dff['Countries'] == value2[2]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]
    return go.Figure(data=go.Scatter(x=dffff['Year'], y=dffff[value1[3]]))


@app.callback(Output('graph4', 'figure'),
              [Input('year_slider', 'value')],
              [Input('well_statuses', 'value')],
              [Input('country_types', 'value')])
def update_figure(value, value1, value2):
    dff = df.loc[:, ['Countries', 'Year', value1[4]]]
    dfff = dff.loc[dff['Countries'] == value2[3]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]
    return go.Figure(data=go.Scatter(x=dffff['Year'], y=dffff[value1[4]]))


# In[ ]:


if __name__ == '__main__':
    app.run_server(debug=False)

# In[ ]:




