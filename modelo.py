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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import os, glob, datetime
from random import randint
import numpy as np
import pathlib
from collections import Counter
from keras.utils import to_categorical
from sklearn import preprocessing
import collections
import glob
from sklearn.utils import resample

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.utils import class_weight
import random
from keras.preprocessing import image
from keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import keras
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

print("Hello")
print(tf.__version__)


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

##################################################################
#LOADING DATA
##################################################################

all_data = pd.read_csv("/Users/luisabrigo/datathon/data/Crash_Details_Table.csv", header=0)


# # =================================================================
# # Drop columns
# # ----------------------------------------------------------------
# all_data = all_data.drop(all_data.columns[0], axis=1)
all_data = all_data.drop(['OBJECTID', 'CRIMEID', 'CCN', 'PERSONID', 'VEHICLEID', "LICENSEPLATESTATE"], axis=1)



print('#',50*"-")


# =================================================================
# Map dependent variable
# ----------------------------------------------------------------

all_data.loc[all_data.FATAL == "N", 'FATAL'] = 0
all_data.loc[all_data.FATAL == "Y", 'FATAL'] = 1
print('#',50*"-")

print(all_data['FATAL'].value_counts())

# plt.hist(all_data['Accident_Severity'])
# plt.show()

# =================================================================
# Fill the missing values
# ----------------------------------------------------------------

# all_data = all_data.drop(all_data.index[100:310000], axis=0)

for col in all_data.columns:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# printing the dataset shape
print("Dataset No. of Rows: ", all_data.shape[0])
print("Dataset No. of Columns: ", all_data.shape[1])
print('#',50*"-")

# =================================================================
# Encoding
# ----------------------------------------------------------------
listsString = ['PERSONTYPE', 'MAJORINJURY', 'MINORINJURY', 'INVEHICLETYPE', 'TICKETISSUED', 'IMPAIRED', "SPEEDING"]

class_le = LabelEncoder()
for i in listsString:
    all_data[i] = class_le.fit_transform(all_data[i])

print('#',50*"-")

# =================================================================
# Split the dataset
# ----------------------------------------------------------------
#2, 3, 9, 10, 14
#

Y = all_data.FATAL
X = all_data.drop('FATAL', axis=1)

print('#',50*"-")



# =================================================================
# Fill the missing values
# ----------------------------------------------------------------

# all_data = all_data.drop(all_data.index[100:310000], axis=0)

for col in all_data.columns:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# printing the dataset shape
print("Dataset No. of Rows: ", all_data.shape[0])
print("Dataset No. of Columns: ", all_data.shape[1])
print('#',50*"-")



# df = df.dropna()
# #Drop year and Countries column
# df = df.drop(labels=df.columns[:2], axis=1)


#split the dataset into a training set and a test set.
#Use the test set in the final evaluation of our models.

train_dataset = all_data.sample(frac=0.8, random_state=0)
test_dataset = all_data.drop(train_dataset.index)

# Have a quick look at the joint distribution of a few pairs of columns from the training set.
# sns.pairplot(train_dataset[['life-expectancyNotFilledFilled', 'gdp-per-capita-NotFilledFilled', 'access-to-electNotFilledFilled', 'population-usinNotFilledFilled']], diag_kind='kde')
# plt.show()

# Also look at the overall statistics, note how each feature covers a very different range:
# print(train_dataset.describe().transpose())


################################################
# TRAIN TEST SPLIT
################################################

#Separate the target value, the "label", from the features. This label is the value that you will train the model to predict.

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('FATAL')
test_labels = test_features.pop('FATAL')

# #%%---------------# =================================================================
# # Resampling Train Set
## SOURCE: https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
# # --------------------------------------------------------------------------------------

# concatenate our training data back together
data = pd.concat([train_features, train_labels], axis=1)

# separate minority and majority classes
no_fatal = data[data.FATAL==0]
fatal = data[data.FATAL==1]

# upsample minority
fatal_upsampled = resample(fatal,
                          replace=True, # sample with replacement
                          n_samples=len(no_fatal), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([fatal_upsampled, no_fatal])

# check new class counts
print(upsampled['FATAL'].value_counts())


print(upsampled.columns)

#Split
train_labels = upsampled.values[:, -1]
train_features = upsampled.drop(['FATAL'], axis=1)
train_features = train_features.values


from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Evaluate model using standardized dataset.
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=10, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, train_features, train_labels, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



from sklearn.metrics import classification_report
import sklearn.metrics as metrics

classes = ['No Fatal','Fatal']

Y_pred = KerasClassifier.predict(test_features)

y_preds = np.argmax(Y_pred, axis=1)

print("---"*10)
print('Confusion Matrix')
print(metrics.confusion_matrix(test_labels, y_preds))
print("")
print('Classification Report')
print(classification_report(test_labels, y_preds,
target_names=classes))
print("---"*10)