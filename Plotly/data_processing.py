import tensorflow as tf
# import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import preprocessing



# def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
#     '''
#     Resizes images keeping proportions.
#     Only a new height or width should be specified: the ratio between the new width or height and the old will be used
#     to scale the other, non-specified dimension.
#     Source: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
#     '''
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     dim = None
#     (h, w) = image.shape[:2]
#
#     # if both the width and height are None, then return the
#     # original image
#     if width is None and height is None:
#         return image
#
#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)
#
#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))
#
#     # resize the image
#     resized = cv2.resize(image, dim, interpolation=inter)
#
#     # return the resized image
#     return resized
#
#
# # -------------------------------------------
# # IMAGE PREPROCESSING
# # -------------------------------------------
#
# os.chdir('..')
# cwd = os.getcwd()
# print(cwd)
#
# # Identify files for images
# folders = [x for x in os.listdir(os.path.join(cwd, 'fakeBoobs/data2')) if ".DS_Store" not in x]
#
#
# all_images = []
# all_labels = []
#
# for subfolder in folders:
#     for img in [x for x in os.listdir(os.path.join(cwd, 'fakeBoobs/data2/', subfolder))if ".DS_Store" not in x]:
#         filename = '{}.jpg'.format(img)
#
#         img = cv2.imread(os.path.join(cwd, 'fakeBoobs/data2/', subfolder, img))
#         img = image_resize(img, 400)  #Keeps Proportions
#
#
#         # plt.imshow(img)
#         # plt.show()
#
#         # if detected_circles is None:
#
#         h, w, f = img.shape
#
#         if h > w:
#
#             x = h - w
#             sub = int(x/2)
#             lower = h - sub
#
#             crop_img = img[sub:lower, :]
#
#             # plt.imshow(crop_img)
#             # plt.show()
#
#             img = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
#
#             # plt.imshow(img)
#             # plt.show()
#
#         elif w > h:
#
#             x = w - h
#             sub = int(x / 2)
#             right = w - sub
#
#             crop_img = img[:, sub:right]
#
#             # plt.imshow(crop_img)
#             # plt.show()
#
#             img = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
#
#             # plt.imshow(img)
#             # plt.show()
#
#
#         else:
#
#             img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
#
#             # plt.imshow(img)
#             # plt.show()
#
#         if subfolder == "real":
#             cv2.imwrite((os.path.join(cwd, 'fakeBoobs/datafinal/final_real', filename)), img)
#         elif subfolder == "fake":
#             cv2.imwrite((os.path.join(cwd, 'fakeBoobs/datafinal/final_fake', filename)), img)
#
