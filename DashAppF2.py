#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
                            max=2020,
                            value=[1990, 2018],
                            className="dcc_control",
                        ),
                        html.Div(id='output-container-range-slider'),

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
                    style={'height': '600px', 'width': '48%', 'display': 'inline-block'}
                ),

                    dcc.Graph(id='graph1', className="pretty_container four columns", style={'height': '600px', 'width': '48%', 'display': 'inline-block'})

            ],
            className="row flex-display",
        ),
        html.Div(
            [
                dcc.Graph(id='graph2', className="pretty_container four columns",
                          style={'height': '600px', 'width': '48%', 'display': 'inline-block'}
                          ),
                dcc.Graph(id='graph3', className="pretty_container four columns",
                          style={'height': '600px', 'width': '48%', 'display': 'inline-block'}
                          )
            ]
        ),

        html.Div(
            [
                dcc.Graph(id='graph4', className="pretty_container four columns",
                          style={'height': '600px', 'width': '48%', 'display': 'inline-block'}
                          ),
                dcc.Graph(id='graph5', className="pretty_container four columns",
                          style={'height': '600px', 'width': '48%', 'display': 'inline-block'}
                          )
            ]
        )

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"}
)


@app.callback(
    dash.dependencies.Output('output-container-range-slider', 'children'),
    [dash.dependencies.Input('year_slider', 'value')])
def update_output(value):
    return 'From "{}"'.format(value[0]), ' to "{}"'.format(value[1])


@app.callback(
    Output('graph1', 'figure'),
    [Input('year_slider', 'value')],
    [Input('well_statuses', 'value')],
    [Input('country_types', 'value')])

def update_figure(value, value1, value2):
    dff = df.loc[:, ['Countries', 'Year', value1[0], value1[1]]]
    dfff = dff.loc[dff['Countries'] == value2[0]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

    if len(value2) > 1:
        dff2 = df.loc[:, ['Countries', 'Year', value1[0], value1[1]]]
        dfff2 = dff2.loc[dff['Countries'] == value2[1]]
        dffff2 = dfff2.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

        if len(value2) > 2:
            dff3 = df.loc[:, ['Countries', 'Year', value1[0], value1[1]]]
            dfff3 = dff3.loc[dff['Countries'] == value2[2]]
            dffff3 = dfff3.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

            if len(value2) > 3:
                dff4 = df.loc[:, ['Countries', 'Year', value1[0], value1[1]]]
                dfff4 = dff4.loc[dff['Countries'] == value2[3]]
                dffff4 = dfff4.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]


    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dffff['Year'], y=dffff[value1[0]], name=str(value2[0][:5]) + " - " + str(value1[0][:10]),
                             line=dict(color='#3cacc6', width=4)), secondary_y=False)
    fig.add_trace(go.Scatter(x=dffff['Year'], y=dffff[value1[1]], name=str(value2[0][:5]) + " - " + str(value1[1][:10]),
                             line=dict(color='#b4e0f2', width=4, dash='dash')), secondary_y=True)
    if len(value2) > 1:
        fig.add_trace(go.Scatter(x=dffff2['Year'], y=dffff2[value1[0]], name=str(value2[1][:5]) + " - " + str(value1[0][:10]),
                                 line=dict(color='#ffce00', width=4)), secondary_y=False)
        fig.add_trace(go.Scatter(x=dffff2['Year'], y=dffff2[value1[1]], name=str(value2[1][:5]) + " - " + str(value1[1][:10]),
                                 line=dict(color='#ffe67d', width=4, dash='dash')), secondary_y=True)

        if len(value2) > 2:
            fig.add_trace(go.Scatter(x=dffff3['Year'], y=dffff3[value1[0]], name=str(value2[2][:5]) + " - " + str(value1[0][:10]),
                                     line=dict(color='#3cce68', width=4)), secondary_y=False)
            fig.add_trace(go.Scatter(x=dffff3['Year'], y=dffff3[value1[1]], name=str(value2[2][:5]) + " - " + str(value1[1][:10]),
                                     line=dict(color='#79c991', width=4, dash='dash')), secondary_y=True)

            if len(value2) > 3:
                fig.add_trace(go.Scatter(x=dffff4['Year'], y=dffff4[value1[0]],
                                         name=str(value2[3][:5]) + " - " + str(value1[0][:10]),
                                         line=dict(color='#f96a4e', width=4)), secondary_y=False)
                fig.add_trace(go.Scatter(x=dffff4['Year'], y=dffff4[value1[1]],
                                         name=str(value2[3][:5]) + " - " + str(value1[1][:10]),
                                         line=dict(color='#fc8b75', width=4, dash='dash')), secondary_y=True)

    fig.update_yaxes(title_text=value1[0][:20], secondary_y=False, showgrid=False, zerolinewidth=0.5, zerolinecolor='rgb(204, 204, 204)')
    fig.update_yaxes(title_text=value1[1][:20], secondary_y=True, showgrid=False, zerolinewidth=0.5, zerolinecolor='rgb(204, 204, 204)')
    fig.update_xaxes(title_text="Year", showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside')

    fig.update_layout(
        title_text="Double Y Axis",
        font_family="Roboto",
        font_color="grey",
        title_font_family="Roboto",
        title_font_color="grey",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig




@app.callback(Output('graph2', 'figure'),
              [Input('year_slider', 'value')],
              [Input('well_statuses', 'value')],
              [Input('country_types', 'value')
               ])
def update_figure(value, value1, value2):
    dff = df.loc[:, ['Countries', 'Year', value1[2]]]
    dfff = dff.loc[dff['Countries'] == value2[0]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

    if len(value2) > 1:
        dff2 = df.loc[:, ['Countries', 'Year', value1[2]]]
        dfff2 = dff2.loc[dff['Countries'] == value2[1]]
        dffff2 = dfff2.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

        if len(value2) > 2:
            dff3 = df.loc[:, ['Countries', 'Year', value1[2]]]
            dfff3 = dff3.loc[dff['Countries'] == value2[2]]
            dffff3 = dfff3.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

            if len(value2) > 3:
                dff4 = df.loc[:, ['Countries', 'Year', value1[2]]]
                dfff4 = dff4.loc[dff['Countries'] == value2[3]]
                dffff4 = dfff4.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dffff['Year'], y=dffff[value1[2]], name=str(value2[0][:15]),
                                 line=dict(color='#3cacc6', width=4)))
    if len(value2) > 1:
        fig.add_trace(go.Scatter(x=dffff2['Year'], y=dffff2[value1[2]], name=str(value2[1][:15]),
                                 line=dict(color='#ffce00', width=4)))
        if len(value2) > 2:
            fig.add_trace(go.Scatter(x=dffff3['Year'], y=dffff3[value1[2]], name=str(value2[2][:15]),
                                 line=dict(color='#3cce68', width=4)))
            if len(value2) > 3:
                fig.add_trace(go.Scatter(x=dffff4['Year'], y=dffff4[value1[2]], name=str(value2[3][:15]),
                                 line=dict(color='#f96a4e', width=4)))

    fig.update_yaxes(title_text=value1[2][:30], showgrid=False, zerolinewidth=0.5, zerolinecolor='rgb(204, 204, 204)')
    fig.update_xaxes(title_text="Year", showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside')

    fig.update_layout(
        title_text="Double Y Axis",
        font_family="Roboto",
        font_color="grey",
        title_font_family="Roboto",
        title_font_color="grey",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )


    return fig


@app.callback(Output('graph3', 'figure'),
              [Input('year_slider', 'value')],
              [Input('well_statuses', 'value')],
              [Input('country_types', 'value')])

def update_figure(value, value1, value2):


    dff = df.loc[:, ['Countries', 'Year', value1[3]]]
    dfff = dff.loc[dff['Countries'] == value2[0]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

    if len(value2) > 1:
        dff2 = df.loc[:, ['Countries', 'Year', value1[3]]]
        dfff2 = dff2.loc[dff['Countries'] == value2[1]]
        dffff2 = dfff2.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

        if len(value2) > 2:
            dff3 = df.loc[:, ['Countries', 'Year', value1[3]]]
            dfff3 = dff3.loc[dff['Countries'] == value2[2]]
            dffff3 = dfff3.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

            if len(value2) > 3:
                dff4 = df.loc[:, ['Countries', 'Year', value1[3]]]
                dfff4 = dff4.loc[dff['Countries'] == value2[3]]
                dffff4 = dfff4.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dffff['Year'], y=dffff[value1[3]], name=str(value2[0][:15]),
                                 line=dict(color='#3cacc6', width=4)))
    if len(value2) > 1:
        fig.add_trace(go.Scatter(x=dffff2['Year'], y=dffff2[value1[3]], name=str(value2[1][:15]),
                                 line=dict(color='#ffce00', width=4)))
        if len(value2) > 2:
            fig.add_trace(go.Scatter(x=dffff3['Year'], y=dffff3[value1[3]], name=str(value2[2][:15]),
                                 line=dict(color='#3cce68', width=4)))
            if len(value2) > 3:
                fig.add_trace(go.Scatter(x=dffff4['Year'], y=dffff4[value1[3]], name=str(value2[3][:15]),
                                 line=dict(color='#f96a4e', width=4)))

    fig.update_yaxes(title_text=value1[3][:30], showgrid=False, zerolinewidth=0.5, zerolinecolor='rgb(204, 204, 204)')
    fig.update_xaxes(title_text="Year", showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside')

    fig.update_layout(
        title_text="Double Y Axis",
        font_family="Roboto",
        font_color="grey",
        title_font_family="Roboto",
        title_font_color="grey",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig



@app.callback(Output('graph4', 'figure'),
              [Input('year_slider', 'value')],
              [Input('well_statuses', 'value')],
              [Input('country_types', 'value')])
def update_figure(value, value1, value2):

    dff = df.loc[:, ['Countries', 'Year', value1[4]]]
    dfff = dff.loc[dff['Countries'] == value2[0]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

    if len(value2) > 1:
        dff2 = df.loc[:, ['Countries', 'Year', value1[4]]]
        dfff2 = dff2.loc[dff['Countries'] == value2[1]]
        dffff2 = dfff2.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

        if len(value2) > 2:
            dff3 = df.loc[:, ['Countries', 'Year', value1[4]]]
            dfff3 = dff3.loc[dff['Countries'] == value2[2]]
            dffff3 = dfff3.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

            if len(value2) > 3:
                dff4 = df.loc[:, ['Countries', 'Year', value1[4]]]
                dfff4 = dff4.loc[dff['Countries'] == value2[3]]
                dffff4 = dfff4.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dffff['Year'], y=dffff[value1[4]], name=str(value2[0][:15]),
                             line=dict(color='#3cacc6', width=4)))
    if len(value2) > 1:
        fig.add_trace(go.Scatter(x=dffff2['Year'], y=dffff2[value1[4]], name=str(value2[1][:15]),
                                 line=dict(color='#ffce00', width=4)))
        if len(value2) > 2:
            fig.add_trace(go.Scatter(x=dffff3['Year'], y=dffff3[value1[4]], name=str(value2[2][:15]),
                                     line=dict(color='#3cce68', width=4)))
            if len(value2) > 3:
                fig.add_trace(go.Scatter(x=dffff4['Year'], y=dffff4[value1[4]], name=str(value2[3][:15]),
                                         line=dict(color='#f96a4e', width=4)))

    fig.update_yaxes(title_text=value1[4][:30], showgrid=False, zerolinewidth=0.5, zerolinecolor='rgb(204, 204, 204)')
    fig.update_xaxes(title_text="Year", showline=True, showgrid=False, showticklabels=True,
                     linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside')

    fig.update_layout(
        title_text="Double Y Axis",
        font_family="Roboto",
        font_color="grey",
        title_font_family="Roboto",
        title_font_color="grey",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


@app.callback(Output('graph5', 'figure'),
              [Input('year_slider', 'value')],
              [Input('well_statuses', 'value')],
              [Input('country_types', 'value')])

def update_figure(value, value1, value2):

    dff = df.loc[:, ['Countries', 'Year', value1[5]]]
    dfff = dff.loc[dff['Countries'] == value2[0]]
    dffff = dfff.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

    if len(value2) > 1:
        dff2 = df.loc[:, ['Countries', 'Year', value1[5]]]
        dfff2 = dff2.loc[dff['Countries'] == value2[1]]
        dffff2 = dfff2.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

        if len(value2) > 2:
            dff3 = df.loc[:, ['Countries', 'Year', value1[5]]]
            dfff3 = dff3.loc[dff['Countries'] == value2[2]]
            dffff3 = dfff3.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

            if len(value2) > 3:
                dff4 = df.loc[:, ['Countries', 'Year', value1[5]]]
                dfff4 = dff4.loc[dff['Countries'] == value2[3]]
                dffff4 = dfff4.loc[(dff['Year'] >= value[0]) & (dff['Year'] <= value[1])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dffff['Year'], y=dffff[value1[5]], name=str(value2[0][:15]),
                             line=dict(color='#3cacc6', width=4)))
    if len(value2) > 1:
        fig.add_trace(go.Scatter(x=dffff2['Year'], y=dffff2[value1[5]], name=str(value2[1][:15]),
                                 line=dict(color='#ffce00', width=4)))
        if len(value2) > 2:
            fig.add_trace(go.Scatter(x=dffff3['Year'], y=dffff3[value1[5]], name=str(value2[2][:15]),
                                     line=dict(color='#3cce68', width=4)))
            if len(value2) > 3:
                fig.add_trace(go.Scatter(x=dffff4['Year'], y=dffff4[value1[5]], name=str(value2[3][:15]),
                                         line=dict(color='#f96a4e', width=4)))

    fig.update_yaxes(title_text=value1[5][:30], showgrid=False, zerolinewidth=0.5, zerolinecolor='rgb(204, 204, 204)')
    fig.update_xaxes(title_text="Year", showline=True, showgrid=False, showticklabels=True,
                     linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside')

    fig.update_layout(
        title_text="Double Y Axis",
        font_family="Roboto",
        font_color="grey",
        title_font_family="Roboto",
        title_font_color="grey",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


# In[ ]:


if __name__ == '__main__':
    app.run_server(debug=False)

# In[ ]:




