import keras
import time
import functools
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import axes3d
# from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from time import gmtime
from time import strftime

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
# from sklearn.metrics import plot_confusion_matrix

num_classes = 40
num_views = 12
img_rows, img_cols = 84,84
channel=1
input_shape = (img_rows, img_cols, channel)

from PIL import Image
import numpy as np
import cv2
import os
my_train = []
my_test = []
y_train=[]
y_test=[]
label={'airplane':np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'bathtub':np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'bed':np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'bench':np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'bookshelf':np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'bottle':np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'bowl':np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'car':np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'chair':np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'cone':np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'cup':np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'curtain':np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'desk':np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'door':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'dresser':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'flower_pot':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'glass_box':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'guitar':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'keyboard':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'lamp':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'laptop':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'mantel':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'monitor':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'night_stand':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'person':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'piano':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),'plant':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),'radio':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),'range_hood':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),'sink':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),'sofa':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),'stairs':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),'stool':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),'table':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),'tent':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),'toilet':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),'tv_stand':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),'vase':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),'wardrobe':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),'xbox':np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])}
train_dir = 'date_max/modelnet40v1_84/train'
valid_dir = 'date_max/modelnet40v1_84/test'
train_dir_list = sorted(os.listdir(train_dir))
valid_dir_list = sorted(os.listdir(valid_dir))

# for conut_num_1 in range(0,len(train_dir_list)):
for conut_num_1 in range(0 + 2 * 0, 0 + 2 * 20):
    train_dir_list_dir = sorted(os.listdir(train_dir + '/' + train_dir_list[conut_num_1]))
    for count_num_2 in range(0, len(train_dir_list_dir)):
        my_train_picture_p = ((np.array(
            (Image.open(str(train_dir + '/' + train_dir_list[conut_num_1]) + '/' + train_dir_list_dir[count_num_2])))))
        my_train_picture_f = keras.utils.normalize(my_train_picture_p, axis=1)
        my_train_picture = my_train_picture_f.astype('float32')
        #         print(my_train_picture)
        my_train.append(my_train_picture)
        y_train.append(label[train_dir_list[conut_num_1]])

add_list = []
flag = 0
for count_num_3 in range(0, len(my_train)):
    add_list.append(my_train[count_num_3])
    if len(add_list) == 12:
        np.save('./date_max/modelnet40v1_84_npy/train/' + 'x_train_' + str(flag) + '.npy',
                (np.array(add_list)).reshape(12, img_rows, img_cols, channel))
        flag = flag + 1
        add_list = []

flag = 0
for count_num_4 in range(0, len(y_train)):
    add_list.append(y_train[count_num_4])
    if len(add_list) == 12:
        np.save('./date_max/modelnet40v1_84_npy/train/' + 'y_train_' + str(flag) + '.npy', (add_list[1]).reshape(40))
        flag = flag + 1
        add_list = []

print(flag)

for conut_num_1 in range(0 + 4 * 0, 0 + 4 * 10):
    valid_dir_list_dir = sorted(os.listdir(valid_dir + '/' + valid_dir_list[conut_num_1]))
    for count_num_2 in range(0, len(valid_dir_list_dir)):
        my_test_picture_p = ((np.array(
            (Image.open(str(valid_dir + '/' + valid_dir_list[conut_num_1]) + '/' + valid_dir_list_dir[count_num_2])))))
        my_test_picture_f = keras.utils.normalize(my_test_picture_p, axis=1)
        my_test_picture = my_test_picture_f.astype('float32')
        my_test.append(my_test_picture)
        y_test.append(label[valid_dir_list[conut_num_1]])

add_list = []
flag = 0
for count_num_3 in range(0, len(my_test)):
    add_list.append(my_test[count_num_3])
    if len(add_list) == 12:
        np.save('./date_max/modelnet40v1_84_npy/test/' + 'x_test_' + str(flag) + '.npy',
                (np.array(add_list)).reshape(12, img_rows, img_cols, channel))
        flag = flag + 1
        add_list = []

flag = 0
for count_num_4 in range(0, len(y_test)):
    add_list.append(y_test[count_num_4])
    if len(add_list) == 12:
        np.save('./date_max/modelnet40v1_84_npy/test/' + 'y_test_' + str(flag) + '.npy', (add_list[1]).reshape(40))
        flag = flag + 1
        add_list = []

print(flag)