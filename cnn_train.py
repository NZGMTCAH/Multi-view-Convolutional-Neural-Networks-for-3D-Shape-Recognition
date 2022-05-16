import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
import time
from tensorflow import keras

# 常量的定义
train_dir = "modelnet40v1/train"
valid_dir = "modelnet40v1/test"
label_file = "labels.txt"
# print(os.path.exists(train_dir))
# print(os.path.exists(valid_dir))
# print(os.path.exists(label_file))
# print(os.listdir(train_dir))
# print(os.listdir(valid_dir))

# 定义常量
height = 84 # resne50的处理的图片大小
width = 84 # resne50的处理的图片大小
channels = 1
batch_size = 4 # 因为处理的图片变大,batch_size变小一点 32->24
num_classes = 40

train_datagen = keras.preprocessing.image.ImageDataGenerator()
valid_datagen = keras.preprocessing.image.ImageDataGenerator()

# 从训练集的文件夹中读取图片
train_generator = train_datagen.flow_from_directory(train_dir,# 图片的文件夹位置
target_size = (height, width),# 将图片缩放到的大小
batch_size = batch_size, # 多少张为一组
seed = 7,#随机数种子
shuffle = True,# 是否做混插
class_mode = "categorical",
color_mode = "grayscale"
) # 控制目标值label的形式-选择onehot编码后的形式
# 从验证集的文件夹中读取图片
valid_generator = valid_datagen.flow_from_directory(valid_dir,
target_size = (height, width),
batch_size = batch_size,
seed = 7,
shuffle = False,
class_mode = "categorical",
color_mode = "grayscale"
)

# 2，查看训练家和验证集分别有多少张数据
train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)

# 3，如何从ImageDataGenerator中读取数据
for i in range(1):
    x, y = train_generator.next()
print(x.shape, y.shape)
print(y)

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Flatten,Dropout,Dense
# 训练模型
# 导入所需工具包
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report,accuracy_score
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
# import utils_paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os


def VGGNet16(width, height, depth, classes):
    model = Sequential()

    model.add(Conv2D(12, (3, 3), input_shape=(width, height, depth), padding='same', activation='relu', name='conv1_block'))
    model.add(Conv2D(12, (3, 3), activation='relu', padding='same', name='conv2_block'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(12*2, (3, 3), activation='relu', padding='same', name='conv3_block'))
    model.add(Conv2D(12*2, (3, 3), activation='relu', padding='same', name='conv4_block'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(12*4, (3, 3), activation='relu', padding='same', name='conv5_block'))
    model.add(Conv2D(12*4, (3, 3), activation='relu', padding='same', name='conv6_block'))
    model.add(Conv2D(12*4, (1, 1), activation='relu', padding='same', name='conv7_block'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(12*8, (3, 3), activation='relu', padding='same', name='conv8_block'))
    model.add(Conv2D(12*8, (3, 3), activation='relu', padding='same', name='conv9_block'))
    model.add(Conv2D(12*8, (1, 1), activation='relu', padding='same', name='conv10_block'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(12*8, (3, 3), activation='relu', padding='same', name='conv11_block'))
    model.add(Conv2D(12*8, (3, 3), activation='relu', padding='same', name='conv12_block'))
    model.add(Conv2D(12*8, (1, 1), activation='relu', padding='same', name='conv13_block'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(40, activation='sigmoid'))
    # model.add(Dense(2, activation='softmax'))
    return model

from keras.models import load_model
model = VGGNet16(width=width, height=height, depth=1, classes=40)
model.summary()

# 设置初始化超参数
INIT_LR = 0.01
EPOCHS = 100
BS = 4

# 损失函数，编译模型
from keras.callbacks import ModelCheckpoint

filepath = 'weights.best.model'
# 有一次提升, 则覆盖一次.
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
callbacks_list = [checkpoint]
from keras.optimizers import SGD

print("------准备训练网络------")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        steps_per_epoch=train_num // batch_size,
                        epochs=EPOCHS,
                        validation_steps=valid_num // batch_size,
                        callbacks=callbacks_list)