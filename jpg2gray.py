import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
import time
import keras
import cv2

# 常量的定义
train_dir = 'date_max/modelnet40v1_84/train'
valid_dir = 'date_max/modelnet40v1_84/test'
img_rows = 84
img_cols = 84


# RGB转灰度并调整大小
from PIL import Image
import numpy as np
import cv2
import os



for conut_num_1 in range(0,len(os.listdir(train_dir))):
    for count_num_2 in range(0,len(os.listdir(train_dir + '/' + os.listdir(train_dir)[conut_num_1]))):
        gray = cv2.cvtColor(np.float32(Image.open((train_dir + '/' + os.listdir(train_dir)[conut_num_1])+'/'+((os.listdir(train_dir + '/' + os.listdir(train_dir)[conut_num_1]))[count_num_2]))),cv2.COLOR_BGR2GRAY)
        gray_re = cv2.resize(gray,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        cv2.imwrite((train_dir + '/' + os.listdir(train_dir)[conut_num_1])+'/'+((os.listdir(train_dir + '/' + os.listdir(train_dir)[conut_num_1]))[count_num_2]), gray_re)

for conut_num_1 in range(0,len(os.listdir(valid_dir))):
    for count_num_2 in range(0,len(os.listdir(valid_dir + '/' + os.listdir(valid_dir)[conut_num_1]))):
        gray = cv2.cvtColor(np.float32(Image.open((valid_dir + '/' + os.listdir(valid_dir)[conut_num_1])+'/'+((os.listdir(valid_dir + '/' + os.listdir(valid_dir)[conut_num_1]))[count_num_2]))),cv2.COLOR_BGR2GRAY)
        gray_re = cv2.resize(gray,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        cv2.imwrite((valid_dir + '/' + os.listdir(valid_dir)[conut_num_1])+'/'+((os.listdir(valid_dir + '/' + os.listdir(valid_dir)[conut_num_1]))[count_num_2]), gray_re)

