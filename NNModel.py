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

print(tf.__version__)


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

##################################################################
#LOADING DATA
##################################################################

df = pd.read_csv("/Users/luisabrigo/HumanProgress/result.csv", header=0, parse_dates=[0])

df = df.dropna()
#Drop year and Countries column
df = df.drop(labels=df.columns[:2], axis=1)


#split the dataset into a training set and a test set.
#Use the test set in the final evaluation of our models.

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

# Have a quick look at the joint distribution of a few pairs of columns from the training set.
sns.pairplot(train_dataset[['life-expectancyNotFilledFilled', 'gdp-per-capita-NotFilledFilled', 'access-to-electNotFilledFilled', 'population-usinNotFilledFilled']], diag_kind='kde')
plt.show()

# Also look at the overall statistics, note how each feature covers a very different range:
print(train_dataset.describe().transpose())


################################################
# TRAIN TEST SPLIT
################################################

#Separate the target value, the "label", from the features. This label is the value that you will train the model to predict.

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('life-expectancyNotFilledFilled')
test_labels = test_features.pop('life-expectancyNotFilledFilled')


################################################
# NORMALIZATION
################################################

# In the table of statistics it's easy to see how different the ranges of each feature are.
print(train_dataset.describe().transpose()[['mean', 'std']])


#The Normalization layer
#The preprocessing.Normalization layer is a clean and simple way to build that preprocessing into your model.

#The first step is to create the layer:
normalizer = preprocessing.Normalization()

#Then .adapt() it to the data:
normalizer.adapt(np.array(train_features))

#This calculates the mean and variance, and stores them in the layer.
print(normalizer.mean.numpy())

#When the layer is called it returns the input data, with each feature independently normalized:
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


################################################
# LINEAR REGRESSION
################################################

#Before building a DNN model, start with a linear regression.
# Start with a single-variable linear regression, to predict Life Exp from GDP Per Capita.
#First create the GDP per Capita Normalization layer:

GDP = np.array(train_features['gdp-per-capita-NotFilledFilled'])

GDP_normalizer = preprocessing.Normalization(input_shape=[1,])
GDP_normalizer.adapt(GDP)


#Build the sequential model:
GDP_model = tf.keras.Sequential([
    GDP_normalizer,
    layers.Dense(units=1)
])

GDP_model .summary()

# This model will predict Life Exp from GDP per Capita.
#Run the untrained model on the first 10 horse-power values.
#The output won't be good, but you'll see that it has the expected shape, (10,1):

GDP_model_predict = GDP_model.predict(GDP[:10])
print(GDP_model_predict)
print(GDP_model_predict.shape)

#Once the model is built, configure the training procedure using the Model.compile() method.
# The most important arguments to compile are the loss and the optimizer
# since these define what will be optimized (mean_absolute_error) and how (using the optimizers.Adam).

GDP_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

#Once the training is configured, use Model.fit() to execute the training:

history = GDP_model.fit(
    train_features['gdp-per-capita-NotFilledFilled'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


#Visualize the model's training progress using the stats stored in the history object.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

#Collect the results on the test set, for later:
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Life Expectancy]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()

test_results = {}

test_results['GDP_model'] = GDP_model.evaluate(
    test_features['gdp-per-capita-NotFilledFilled'],
    test_labels, verbose=0)

#Since this is a single variable regression it's easy to look at the model's predictions
# as a function of the input:

x = tf.linspace(0.0, 250, 251)
y = GDP_model.predict(x)

def plot_GDP(x, y):
  plt.scatter(train_features['gdp-per-capita-NotFilledFilled'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('gdp-per-capita-NotFilledFilled')
  plt.ylabel('life-expectancyNotFilledFilled')
  plt.legend()

plot_GDP(x,y)
plt.show()


################################################
# FULL DNN MODEL
################################################

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
dnn_model.summary()


history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)
plt.show()

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

################################################
# PERFORMANCE
################################################

perf = pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
print(perf)

################################################
# PREDICTIONS
################################################

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [LifeExp]')
plt.ylabel('Predictions [LifeExp]')
lims = [40, 100]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [LifeExp]')
_ = plt.ylabel('Count')
plt.show()