import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy.linalg as LA
import statsmodels.api as sm
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

np.set_printoptions(precision=3, suppress=True)


# Models
#1. Multiple Linear Regression using statsmodel
#2. DNN Regression using Tensorflow

##################################################################
#LOADING DATA
##################################################################

df = pd.read_csv("/Users/luisabrigo/humanprogressorg/result.csv", header=0, parse_dates=[0])
df = df.dropna()

df_Y = df["life-expectancy"]

df_X = df.drop("life-expectancy", 1)
df_X = df_X.drop(labels=df.columns[:2], axis=1)

print(df_Y)

##################################################################
#TRAIN TEST SPLIT
##################################################################

Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_X, df_Y, shuffle=True, test_size=0.2)

################################################
#EDA
################################################

#Seaborn Heatmap
import seaborn as sns
corr = df_X.corr()
corr = df.corr()

plt.subplots(figsize=(15,15))
sns.heatmap(corr.T, annot=True, cmap="YlGnBu")
plt.tight_layout(pad=2, w_pad=0.5, h_pad=0.5)
plt.title("Correlation Matrix")
plt.show()

################################################
# COLINEARITY DETECTION
################################################

##Collinearity detection:
#Converts dataframe to numpy matrix
X = Xtrain.values
print("Shape of X is:", X.shape)

#Matrix multiplication
H = np.matmul(X.T, X)
print("Shape of H is:", H.shape)

# Single Value Decomposition. If values are close to zero means the variables are correlated (multicolinearity)
s, d, v = np.linalg.svd(H)
print("Singular values:", d)

#Condition Number
print("Condition number for X is:", LA.cond(X))
print("Colinearity is Severe")



################################################
# FEATURE SELECTION
################################################

# Adds a column called "const"
Xtrain = sm.add_constant(Xtrain)
Xtest = sm.add_constant(Xtest)

#Using python, statsmodels package and OLS function, find the unknown coefficients. C

model = sm.OLS(Ytrain.astype(float), Xtrain.astype(float)).fit()
print(model.summary())

#Feature selection: Using a backward stepwise regression, reduce the feature space
# dimension. You need to use the AIC, BIC and Adjusted R2 as a predictive accuracy for your analysis.

# remove economically-acNotFilledFilled
x_opt = Xtrain.drop("child-labor", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())

# # remove human-developmeNotFilledFilled
x_opt = x_opt.drop("hdi", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())

# # remove share-of-peopleNotFilledFille
x_opt = x_opt.drop("life-satisfaction", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())

# # remove co2-emissions-pNotFilledFilled
x_opt = x_opt.drop("co2-emissions", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())

# # remove gdp-annual-growNotFilledFilled
x_opt = x_opt.drop("gdp-annual-growth", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())


################################################
# MULTIPLE LINEAR REGRESSION
################################################

# Edit the test set
Xtest_new = Xtest[["const", "vaccination",
    "improved-water-sources", "homicide-rate",
    "gdp-per-capita", "mean-years-of-schooling",
    "child-mortality", "access-to-electricity", "poverty"]]


Ypred = ols.predict(x_opt)
Yforecast = ols.predict(Xtest_new)
print(Yforecast)


################################################
# RESULTS
################################################

test_results = {}

def variance(x, k):
    var = np.sqrt((1/(len(x)-k-1))*(np.sum(np.square(x))))
    return var

#Calculates prediciton and forecast error
p_error = Ytrain - Ypred
print("This is the prediction error variance:", variance(p_error, 4))

f_error = Ytest - Yforecast
print("This is the forecast error variance:", variance(f_error, 4))


def MSE(x):
    return np.sum(np.power(x, 2)) / len(x)

MSE = MSE(f_error)
print("MSE Prediction Error: ", round(MSE, 3))

# Calculate R2
def R2(yhat, y_test):
    y_mean = np.mean(y_test)
    r2 = np.sum((yhat - y_mean) ** 2) / np.sum((y_test - y_mean) ** 2)
    return r2


R2 = R2(Yforecast, Ytest)
print("The R^2 coefficient is:", round(R2, 3))


a = plt.axes(aspect='equal')
plt.scatter(Ytest, Yforecast)
plt.xlabel('True Values [LifeExp]')
plt.ylabel('Predictions [LifeExp]')
lims = [40, 100]
plt.title("True vs. Predicted values[LifeExp] - Multiple Linear Regression")
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = Ytest - Yforecast
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [LifeExp]')
plt.title("Residuals Error Histogram - Multiple Linear Regression")
_ = plt.ylabel('Count')
plt.show()

test_results['Linear Regression'] = MSE



##################################################################
# DEEP NEURAL NETWORK REGRESSION USING TENSORFLOW
##################################################################

##################################################################
#LOADING DATA
##################################################################

df = pd.read_csv("/Users/luisabrigo/humanprogressorg/result.csv", header=0, parse_dates=[0])
df = df.dropna()
#Drop year and Countries column
df = df.drop(labels=df.columns[:2], axis=1)

for i in df.columns:
    print(i)

# plt.figure(figsize=(15, 15))
# sns.pairplot(df[['life-expectancy', 'gdp-per-capita', 'improved-water-sources', "vaccination", "homicide-rate", "mean-years-of-schooling", "child-mortality", "access-to-electricity", "co2-emissions", "hdi", "gdp-annual-growth", "life-satisfaction", "child-labor", "poverty" ]], diag_kind='kde')
# plt.show()

plt.figure(figsize=(10, 10))
sns.pairplot(df[['life-expectancy', 'gdp-per-capita', "gdp-annual-growth", "vaccination", "child-mortality",  "hdi", ]], diag_kind='kde')
# plt.title("Scatter Plot - Selected Variables")
plt.show()


plt.figure(figsize=(10, 10))
sns.pairplot(df[['life-expectancy', "access-to-electricity", 'improved-water-sources', "poverty", "child-labor"]], diag_kind='kde')
# plt.title("Scatter Plot - Selected Variables")
plt.show()

plt.figure(figsize=(10, 10))
sns.pairplot(df[['life-expectancy', "co2-emissions", "mean-years-of-schooling", "homicide-rate","life-satisfaction"]], diag_kind='kde')
# plt.title("Scatter Plot - Removed Variables")
plt.show()


df = df[['life-expectancy',"vaccination",
    "improved-water-sources", "homicide-rate",
    "gdp-per-capita", "mean-years-of-schooling",
    "child-mortality", "access-to-electricity", "poverty"]]

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)


################################################
# TRAIN TEST SPLIT
################################################

train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('life-expectancy')
test_labels = test_features.pop('life-expectancy')

################################################
# NORMALIZATION
################################################

print(train_dataset.describe().transpose()[['mean', 'std']])

#The Normalization layer

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())



#Collect the results on the test set, for later:
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 5000])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Life Expectancy]')
  plt.title("Error vs. Epochs")
  plt.legend()
  plt.grid(True)


################################################
# FULL DNN MODEL
################################################

def model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


dnn_model = model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)
plt.show()



################################################
# PREDICTIONS
################################################

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [LifeExp]')
plt.ylabel('Predictions [LifeExp]')
plt.title("True vs. Predicted values[LifeExp] - Deep Neural Network")
lims = [40, 100]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [LifeExp]')
plt.title("Residuals Error Histogram - Deep Neural Network")
_ = plt.ylabel('Count')
plt.show()

test_results['Deep Neural Network'] = dnn_model.evaluate(test_features, test_labels, verbose=0)



################################################
# PERFORMANCE
################################################

perf = pd.DataFrame(test_results, index=['Mean Squared Error [MSE]']).T
print(perf)
