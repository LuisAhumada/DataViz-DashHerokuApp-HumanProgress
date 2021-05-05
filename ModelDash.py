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

# pd.set_option('display.max_rows', None)

#Model function applied to Heroku app: humanprogressorg.herokuapp.com
sns.set()

def LinReg(y, x1, x2, *kwargs):
    df = pd.read_csv("/Users/luisabrigo/HumanProgress/result_all_indicators_data2.csv", header=0)

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

    # plt.subplots(figsize=(12,8))
    # sns.heatmap(corr.T, annot=True, cmap="YlGnBu", linewidths=.5)
    # plt.title("Correlation Matrix")
    # plt.tight_layout()
    # plt.savefig('Images/image1.png')
    # plt.show()

    # count = 2
    # for column in df_X:
    #     plt.figure(figsize=(8, 8))
    #     sns.scatterplot(data=df_X, x=column, y=df_a[y], s=20, alpha=0.5)
    #     plt.title("Dependent vs. " + str(column))
    #     plt.tight_layout()
    #     plt.savefig("Images/image" + str(count) + ".png")
    #     plt.show()
    #     count += 1

    #Using python, statsmodels package and OLS function, find the unknown coefficients. C

    # Adds a column called "const"
    Xtrain = sm.add_constant(Xtrain)
    Xtest = sm.add_constant(Xtest)

    model = sm.OLS(Ytrain.astype(float), Xtrain.astype(float)).fit()
    summary = model.summary()
    print(type(summary))

    Ypred = model.predict(Xtest)

    from sklearn.metrics import mean_squared_error

    MSE = mean_squared_error(Ytest, Ypred)

    # plt.figure(figsize=(8, 8))
    # sns.regplot(x=Ytest, y=Ypred)
    # plt.xlabel('True Values ' + str(y))
    # plt.ylabel('Predictions ' + str(y))
    # lims = [min(Ypred), max(Ypred)]
    # plt.xlim(lims)
    # plt.ylim(lims)
    # _ = plt.plot(lims, lims)
    # plt.title("True vs. Predicted values Linear Regression")
    # plt.tight_layout()
    # plt.savefig("Images/image" + str(count) + ".png")
    # count += 1
    # plt.show()

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

    # def plot_loss(history):
    #     plt.figure(figsize=(8, 8))
    #     plt.plot(history.history['loss'], label='loss')
    #     plt.plot(history.history['val_loss'], label='val_loss')
    #     plt.ylim([0, 10])
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Error')
    #     plt.title("Error vs. Epochs")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("Images/image" + str(count) + ".png")
    #     plt.grid(True)

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

    # plot_loss(history)
    # count += 1
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


    # test_predictions = dnn_model.predict(test_features).flatten()
    #
    # plt.figure(figsize=(8, 8))
    # a = plt.axes(aspect='equal')
    # sns.regplot(test_labels, test_predictions)
    # plt.xlabel('True Values ' + str(y))
    # plt.ylabel('Predictions' + str(y))
    # lims = [min(test_predictions), max(test_predictions)]
    # plt.xlim(lims)
    # plt.ylim(lims)
    # _ = plt.plot(lims, lims)
    # plt.title("True vs. Predicted values neural network")
    # plt.tight_layout()
    # plt.savefig("Images/image" + str(count) + ".png")
    # count += 1
    # plt.show()
    #
    #
    # error = test_predictions - test_labels
    # plt.figure(figsize=(8, 8))
    # plt.hist(error, bins=25)
    # plt.xlabel('Prediction Error')
    # _ = plt.ylabel('Count')
    # plt.title("Count of prediction error")
    # plt.tight_layout()
    # plt.savefig("Images/image" + str(count) + ".png")
    # plt.show()

    text1 = summary
    text2 = MSE

    # print("-" * 50)
    # print("-" * 50)
    # print("-" * 50)
    #
    #
    print(text1)
    print("#" * 50)
    print(text2)
    print("#" * 50)
    print(text3, dnn_model.summary())
    print("#" * 50)
    print(text4)



    return text1, text2, text4



Accessed_text1, Accessed_text2, Accessed_text3 = LinReg("Life Expectancy At Birth Men Years 1960–2015",
    "Motorcyclist Road Injury Prevalence Per 100000 People 1990–2017",
    "Conflict And Terrorism Prevalence Per 100000 People 1990–2017", "Cereals Net Production Relative To 2004 2006 1961–2014",
   "Asthma Prevalence Per 100000 People 1990–2017")


print(Accessed_text1)
print(Accessed_text2)

