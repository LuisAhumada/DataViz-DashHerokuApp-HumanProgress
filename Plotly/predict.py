import numpy as np
from keras.models import load_model
# import cv2
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()

# def predict_function(x):
#     # %% --------------------------------------------- Data Prep -------------------------------------------------------
#     img = cv2.imread(x)
#
#     h, w, f = img.shape
#     if h > w:
#         x = h - w
#         sub = int(x / 2)
#         lower = h - sub
#         crop_img = img[sub:lower, :]
#         img = cv2.resize(crop_img, (200, 200), interpolation=cv2.INTER_AREA)
#         img = np.reshape(img, [1, 200, 200, 3])
#         # plt.imshow(img)
#         # plt.show()
#
#     elif w > h:
#         x = w - h
#         sub = int(x / 2)
#         right = w - sub
#         crop_img = img[:, sub:right]
#         img = cv2.resize(crop_img, (200, 200), interpolation=cv2.INTER_AREA)
#         img = np.reshape(img, [1, 200, 200, 3])
#         # plt.imshow(img)
#         # plt.show()
#
#     else:
#         img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
#         img = np.reshape(img, [1, 200, 200, 3])
#         # plt.imshow(img)
#         # plt.show()
#
#     # %% --------------------------------------------- Predict ---------------------------------------------------------
#     model = load_model(cwd + '/fakeBoobs/model.hdf5')
#
#     y_pred = model.predict(img)
#     y_preds = np.argmax(y_pred, axis=1)
#
#     return y_preds
