import os, glob, datetime
from random import randint
import numpy as np
import pathlib
from collections import Counter
from keras.utils import to_categorical
from sklearn import preprocessing
import collections
import glob

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



# load truncated images: https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
# from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

tf.random.set_seed(42)
np.random.seed(42)

cwd = os.getcwd()
dir_train = '/Users/glosophy/Dropbox/fakeBoobs/datasplit/train'
data_train = pathlib.Path(dir_train)

dir_val = '/Users/glosophy/Dropbox/fakeBoobs/datasplit/validation'
data_val = pathlib.Path(dir_val)

dir_test = '/Users/glosophy/Dropbox/fakeBoobs/datasplit/test'
data_test = pathlib.Path(dir_test)
print ("The current working directory is %s" % cwd)

print("---"*10)
print("Starting images and label pre-processing...")


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        data_train,
        target_size=(200, 200),
        batch_size=64,
        class_mode='categorical',
        )

validation_generator = data_generator.flow_from_directory(
    data_val,
    target_size=(200, 200),
    batch_size=64,
    class_mode='categorical',
    )

test_generator = data_generator.flow_from_directory(
    data_test,
    target_size=(200, 200),
    batch_size=64,
    class_mode='categorical',
    )

print("Images and labels successfully pre-processed!")
print("---"*10)


from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

num_classes = 2
model = tf.keras.Sequential()  # Initialize model

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(200,200,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))       # Reduces output dimensions

# SECOND KERAS LAYER:
# Create the second layer with 64 filters (increasing) and a 3x3 kernel size since the images are relatively small
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# THIRD KERAS LAYER:
# Create the third layer with 128 filters (increasing) and a 3x3 kernel size since the images are relatively small
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# fully-connected layer flattening the data and performing a final dense layer
model.add(Flatten())
model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2, seed=42))
model.add(BatchNormalization())

# Adds a final output layer with softmax to map to the 4 classes
model.add(Dense(num_classes, activation="sigmoid"))

print(model.summary())

adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


nb_train_samples = 461
nb_validation_samples = 56
epochs = 20
batch_size = 64

history = model.fit(train_generator, epochs=epochs,
                    batch_size = batch_size,
                    validation_data=validation_generator)


model.save("model.hdf5")


# visualize the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()


from sklearn.metrics import classification_report
import sklearn.metrics as metrics

classes = ['fake','real']

Y_pred = model.predict_generator(test_generator, 81 // 64+1)

y_preds = np.argmax(Y_pred, axis=1)

print("---"*10)
print('Confusion Matrix')
print(metrics.confusion_matrix(test_generator.classes, y_preds))
print("")
print('Classification Report')
print(classification_report(test_generator.classes, y_preds,
target_names=classes))
print("---"*10)