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

batch_size = 5
num_classes = 40
num_views = 12
img_rows, img_cols = 84,84
channel=1
input_shape = (img_rows, img_cols, channel)

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout, Dense
# 训练模型
# 导入所需工具包
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
# import utils_paths
import matplotlib.pyplot as plt
import numpy as np
import random
# import cv2
import os
from keras.models import load_model


class SplitLayer(keras.layers.Layer):
    def __init__(self, num_splits, **kwargs):
        self.num_splits = num_splits
        super(SplitLayer, self).__init__(**kwargs)

    # Defines the computation from inputs to outputs.
    def call(self, x):
        return [x[:, i] for i in range(self.num_splits)]

    # Computes the output shape of the layer.
    def compute_output_shape(self, input_shape):
        return [(input_shape[0],) + input_shape[2:]] * self.num_splits

    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        config['num_splits'] = self.num_splits
        #         config['param2'] = self.param2
        return config


def get_test_shared_model():
    num_channels = 12  # for example
    depth = 1
    model_pre = load_model('vgg12^3-80^2-84-81625.model')
    model = Sequential()
    #     model = model_pre

    for layer in model_pre.layers[:-2]:  # 最后一层
        model.add(layer)

    #     for layer in model.layers:
    #         layer.trainable = False
    #     model.add(Conv2D(num_channels, (3, 3), input_shape=(width, height, depth), padding='same', activation='relu', name='conv1_block'))
    #     model.add(Conv2D(num_channels, (3, 3), activation='relu', padding='same', name='conv2_block'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #     model.add(Conv2D(num_channels*2, (3, 3), activation='relu', padding='same', name='conv3_block'))
    #     model.add(Conv2D(num_channels*2, (3, 3), activation='relu', padding='same', name='conv4_block'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #     model.add(Conv2D(num_channels*4, (3, 3), activation='relu', padding='same', name='conv5_block'))
    #     model.add(Conv2D(num_channels*4, (3, 3), activation='relu', padding='same', name='conv6_block'))
    #     model.add(Conv2D(num_channels*4, (1, 1), activation='relu', padding='same', name='conv7_block'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #     model.add(Conv2D(num_channels*8, (3, 3), activation='relu', padding='same', name='conv8_block'))
    #     model.add(Conv2D(num_channels*8, (3, 3), activation='relu', padding='same', name='conv9_block'))
    #     model.add(Conv2D(num_channels*8, (1, 1), activation='relu', padding='same', name='conv10_block'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #     model.add(Conv2D(num_channels*8, (3, 3), activation='relu', padding='same', name='conv11_block'))
    #     model.add(Conv2D(num_channels*8, (3, 3), activation='relu', padding='same', name='conv12_block'))
    #     model.add(Conv2D(num_channels*8, (1, 1), activation='relu', padding='same', name='conv13_block'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #     model.add(Flatten())
    #     model.add(Dense(4096, activation='relu'))
    #     # model.add(BatchNormalization())
    #     model.add(Dropout(0.5))
    #     model.add(Dense(4096, activation='relu'))
    #     # model.add(BatchNormalization())
    #     model.add(Dropout(0.5))
    #     model.add(Dense(classes, activation='softmax'))
    return model

# num_channels = 12 # for example
# from keras.models import load_model
# cnn = load_model('res84-80083.model')

# cnn.summary()


def test_split_layer():
    num_channels = 12
    num_views = 12  # or any other number
    cnn = get_test_shared_model()
    input = keras.layers.Input((num_views, img_rows, img_cols, channel))
    views = SplitLayer(num_views)(input)  # list of keras-tensors

    processed_views = []  # empty list
    for view in views:
        x = cnn(view)
        processed_views.append(x)
    pooled_views = keras.layers.Maximum()(processed_views)

    #     x = Conv2D(num_channels, (3, 3), activation='relu', padding='same', name='conv2_block')(pooled_views)
    #     x = BatchNormalization()(pooled_views)
    #     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pooled_views)

    #     x = Conv2D(num_channels*2, (3, 3), activation='relu', padding='same', name='conv3_block')(x)
    #     x = Conv2D(num_channels*2, (3, 3), activation='relu', padding='same', name='conv4_block')(x)
    #     x = BatchNormalization()(x)
    #     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    #     x = Conv2D(num_channels*4, (3, 3), activation='relu', padding='same', name='conv5_block')(x)
    #     x = Conv2D(num_channels*4, (3, 3), activation='relu', padding='same', name='conv6_block')(x)
    #     x = Conv2D(num_channels*4, (1, 1), activation='relu', padding='same', name='conv7_block')(x)
    #     x = BatchNormalization()(x)
    #     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    #     x = Conv2D(num_channels*8, (3, 3), activation='relu', padding='same', name='conv8_block')(x)
    #     x = Conv2D(num_channels*8, (3, 3), activation='relu', padding='same', name='conv9_block')(x)
    #     x = Conv2D(num_channels*8, (1, 1), activation='relu', padding='same', name='conv10_block')(x)
    #     x = BatchNormalization()(x)
    #     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    #     x = Conv2D(num_channels*8, (3, 3), activation='relu', padding='same', name='conv11_block')(x)
    #     x = Conv2D(num_channels*8, (3, 3), activation='relu', padding='same', name='conv12_block')(x)
    #     x = Conv2D(num_channels*8, (1, 1), activation='relu', padding='same', name='conv13_block')(x)
    #     x = BatchNormalization()(x)
    #     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    #     x = Flatten()(x)
    #     x = Dense(4096, activation='relu')(x)
    #     # model.add(BatchNormalization())
    #     x = Dropout(0.5)(x)
    #     x = Dense(4096, activation='relu')(pooled_views)
    # model.add(BatchNormalization())
    x = Dropout(0.5)(pooled_views)
    x = Dense(classes, activation='softmax')(x)
    model = keras.models.Model(input, x)
    return model

from keras.models import load_model
width=img_rows
height=img_rows
classes=40
model = test_split_layer()
model.summary()

x_train=np.load("x_train_84.npy")
x_valid=np.load("x_valid_84.npy")
y_train=np.load("y_train_84.npy")
y_valid=np.load("y_valid_84.npy")

from keras.optimizers import SGD

epochs = 100
INIT_LR = 0.01
opt = SGD(lr=INIT_LR, decay=INIT_LR / epochs)
from keras.callbacks import ModelCheckpoint

filepath = 'weights.best.model'
# 有一次提升, 则覆盖一次.
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
callbacks_list = [checkpoint]

model.compile(loss="categorical_crossentropy",
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=callbacks_list)

predictions = model.predict(x_valid,batch_size=batch_size, verbose=0)
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])