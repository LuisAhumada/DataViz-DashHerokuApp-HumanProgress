#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dash
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
from plotly.subplots import make_subplots

import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy.linalg as LA
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy.linalg as LA
import statsmodels.api as sm
# Use seaborn for pairplot
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
pd.set_option('display.max_columns', None)
import seaborn as sns






df = pd.read_csv('/Users/luisabrigo/humanprogress/Viz/result_all_indicators_data3.csv')



df_indicators = list(df.columns.drop(['Unnamed: 0', "Countries", 'Year']))

image_filename = 'Viz/Images/image1.png'  # replace with your own image
encoded_image = base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image1.png', 'rb').read())


sns.set()

def LinReg(y, x1, x2, *kwargs):
    df = pd.read_csv("/Users/luisabrigo/HumanProgress/Viz/result_all_indicators_data3.csv", header=0)
    # df = pd.read_excel('df_all.xlsx')

    arguments = []
    arguments.extend([y, x1, x2])
    for i in kwargs:
        arguments.append(i)
    # print(arguments)

    df_a = pd.DataFrame()
    for j in arguments:
        data = df[[j]]
        # print(data)
        df_a = pd.concat([df_a, data], axis=1)

    df_a = df_a.dropna()
    df_Y = df_a[[y]]
    df_X = df_a.drop(y, 1)

    #Train-test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_X, df_Y, shuffle=True, test_size=0.2)

    #Seaborn Heatmap
    corr = df_a.corr()

    plt.subplots(figsize=(12,8))
    sns.heatmap(corr.T, annot=True, cmap="YlGnBu", linewidths=.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig('Images/image1.png')
    plt.show()

    count = 2
    for column in df_X:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_X, x=column, y=df_a[y], s=20, alpha=0.5)
        plt.title("Dependent vs. " + str(column))
        plt.tight_layout()
        plt.savefig("Images/image" + str(count) + ".png")
        plt.tight_layout()
        plt.show()
        count += 1

    #Using python, statsmodels package and OLS function, find the unknown coefficients. C

    # Adds a column called "const"
    Xtrain = sm.add_constant(Xtrain)
    Xtest = sm.add_constant(Xtest)

    model = sm.OLS(Ytrain.astype(float), Xtrain.astype(float)).fit()
    summary = model.summary()

    Ypred = model.predict(Xtest)

    from sklearn.metrics import mean_squared_error

    MSE = mean_squared_error(Ytest, Ypred)

    plt.figure(figsize=(8, 8))
    sns.regplot(x=Ytest, y=Ypred)
    plt.xlabel('True Values ' + str(y))
    plt.ylabel('Predictions ' + str(y))
    lims = [min(Ypred), max(Ypred)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.title("True vs. Predicted values Linear Regression")
    plt.tight_layout()
    plt.savefig("Images/image" + str(count) + ".png")
    count += 1
    plt.show()

    #DNN
    train_dataset = df_a.sample(frac=0.8, random_state=0)
    test_dataset = df_a.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(y)
    test_labels = test_features.pop(y)

    ################################################
    # NORMALIZATION
    ################################################

    # In the table of statistics it's easy to see how different the ranges of each feature are.
    # print(train_dataset.describe().transpose()[['mean', 'std']])

    # The Normalization layer
    # The preprocessing.Normalization layer is a clean and simple way to build that preprocessing into your model.

    # The first step is to create the layer:
    normalizer = preprocessing.Normalization()

    # Then .adapt() it to the data:
    normalizer.adapt(np.array(train_features))

    # This calculates the mean and variance, and stores them in the layer.
    # print(normalizer.mean.numpy())

    # When the layer is called it returns the input data, with each feature independently normalized:
    first = np.array(train_features[:1])

    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print()
        print('Normalized:', normalizer(first).numpy())

    def plot_loss(history):
        plt.figure(figsize=(8, 8))
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title("Error vs. Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Images/image" + str(count) + ".png")
        plt.grid(True)

    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    dnn_model = build_and_compile_model(normalizer)
    text3 = dnn_model.summary()

    history = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=0, epochs=100)

    plot_loss(history)
    count += 1
    plt.show()

    test_results = {}
    test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

    ################################################
    # PERFORMANCE
    ################################################

    perf = pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
    print(perf)

    text4 = perf
    ################################################
    # PREDICTIONS
    ################################################


    test_predictions = dnn_model.predict(test_features).flatten()

    plt.figure(figsize=(8, 8))
    a = plt.axes(aspect='equal')
    sns.regplot(test_labels, test_predictions)
    plt.xlabel('True Values ' + str(y))
    plt.ylabel('Predictions' + str(y))
    lims = [min(test_predictions), max(test_predictions)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.title("True vs. Predicted values neural network")
    plt.tight_layout()
    plt.savefig("Images/image" + str(count) + ".png")
    count += 1
    plt.show()


    error = test_predictions - test_labels
    plt.figure(figsize=(8, 8))
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error')
    _ = plt.ylabel('Count')
    plt.title("Count of prediction error")
    plt.tight_layout()
    plt.savefig("Images/image" + str(count) + ".png")
    plt.show()

    text1 = summary
    text1 = str(text1)
    text2 = MSE

    print("-" * 50)
    print("-" * 50)
    print("-" * 50)


    print(text1)
    print("#" * 50)
    print(text2)
    print("#" * 50)
    print(text3, dnn_model.summary())
    print("#" * 50)
    print(text4)


    return text1, text2, dnn_model.summary(), text4


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
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.P("Select Dependent Variable:", className="control_label"),
                        dcc.RadioItems(
                            id="dependent_variable_selector",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="dependent_variable",
                            options= [{'label': i, 'value': i} for i in df_indicators],
                            #value=list(INDICATORS.keys()),
                            className="dcc_control",
                        ),
                        html.P("Select Independent Variables:", className="control_label"),
                        dcc.RadioItems(
                            id="Independent_veriable_selector",
                            value="Chile",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="independent_variables",
                            options=[{'label': j, 'value': j} for j in df_indicators],
                            multi=True,
                            #value=list(COUNTRIES.values()),
                            className="dcc_control",
                        ),

                        html.Button('Run Model', id='btn'),
                            html.Div(id='container-button-basic',children=''),



                        dcc.Checklist(id='checklist',options=[{'label': 'See Model Details', 'value': 'SMD'}],)


                    ],
                    className="pretty_container four columns",
                    id="veriable-options",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "50%",
                        "float":"none",
                    },
                )]),




        html.Div(

            [
                html.Img(
                    id="image1",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),

                html.Img(
                    id="image2",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),

                html.Img(
                    id="image3",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),

                html.Img(
                    id="image4",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),

                html.Img(
                    id="image5",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),

                html.Div(id='text1', style={'marginBottom': "50", 'marginTop': "25"}

                ),

                html.Img(
                    id="image6",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),

                # html.Div([
                #     html.P(id='text2')
                # ], style={'marginBottom': "50", 'marginTop': "25"}
                #
                # ),
                #
                # html.Div([
                #     html.P(id='text3')
                # ], style={'marginBottom': "50", 'marginTop': "25"}
                #
                # ),

                html.Img(
                    id="image7",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),

                html.Img(
                    id="image8",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),

                # html.Div([
                #     html.P(id='text4')
                # ], style={'marginBottom': "50", 'marginTop': "25"}
                #
                # ),


                html.Img(
                    id="image9",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "95%",
                        "float":"none",
                    },
                ),


                html.Div(
                    id="profile-about-section",
                    style={
                        "padding-left": "5%",
                        "padding-top": "10%",
                    },
                ),
            ],
            className="pretty_container four columns",
            style={
                "display": "block",
                "margin-left": "auto",
                "margin-right": "auto",
                "width": "50%",
                "float":"none",
            }
        ),


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

@app.callback(Output('container-button-basic', 'children'),
              Output("image1", "src"),
              Output("image2", "src"),
              Output("image3", "src"),
              Output("image4", "src"),
              Output("image5", "src"),
              Output("text1", "children"),
              Output("image6", "src"),
              # Output("text2", "children"),
              # Output("text3", "children"),
              Output("image7", "src"),
              Output("image8", "src"),
              # Output("text4", "children"),
              Output("image9", "src"),
              [Input('dependent_variable', 'value')],
              [Input('independent_variables', 'value')],
              [Input('btn', 'n_clicks')])

def button_click(value, value1, n_clicks):
    text1, text2, text3, text4 = LinReg(value, value1[0], value1[1], value1[2], value1[3])
    return 'model completed', 'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image1.png', 'rb').read()).decode()),\
           'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image2.png', 'rb').read()).decode()),\
           'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image3.png', 'rb').read()).decode()),\
           'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image4.png', 'rb').read()).decode()),\
           'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image5.png', 'rb').read()).decode()),\
           text1,\
           'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image6.png', 'rb').read()).decode()),\
           'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image7.png', 'rb').read()).decode()),\
           'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image8.png', 'rb').read()).decode()),\
           'data:image/png;base64,{}'.format(base64.b64encode(open('/Users/luisabrigo/HumanProgress/Viz/Images/image9.png', 'rb').read()).decode())













if __name__ == '__main__':
    app.run_server(debug=False)

# In[ ]:


# @app.callback(Output("body-image", "src"),
#               [Input("weight-plot", "hoverData")])
# def update_body_image(hover_data):
#     date = hover_data["points"][0]["x"]
#     src = "/assets/{}.jpg".format(date)
#
#     return src


