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
# import cv2
import os
from keras.models import load_model

import keras


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

model = keras.models.load_model('MVCNN-87125.model',custom_objects={'SplitLayer': SplitLayer})

import random
num=39
Xnew = (np.load("./date_max/modelnet40v1_84_npy/test/x_test_"+str(random.randint(0+20*num,19+20*num))+".npy")).reshape(1,12, 84, 84,1)
predictions = (model.predict(Xnew)).tolist()
pre_p = predictions[0]

pre_p_str = []
for count_num in range(0,len(pre_p)):
    pre_p_str.append(str(pre_p[count_num]))

(predictions[0]).sort(reverse=True)
pre_s = predictions[0]

pre_s_str = []
for count_num in range(0,len(pre_s)):
    pre_s_str.append(str(pre_s[count_num]))

p_1=(pre_p_str).index(pre_s_str[0])
p_2=(pre_p_str).index(pre_s_str[1])
p_3=(pre_p_str).index(pre_s_str[2])
label = {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf', 5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone', 10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser', 15: 'flowerpot', 16: 'glassbox', 17: 'guitar', 18: 'keyboard', 19: 'lamp', 20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'nightstand', 24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'rangehood', 29: 'sink', 30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet', 36: 'tvstand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'}

# name_list = [label[p_1],label[p_2],label[p_3]]
# print(name_list)

import matplotlib.pyplot as plt


name_list = [label[p_1],label[p_2],label[p_3]]
num_list = [pre_s[0],pre_s[1],pre_s[2]]

plt.barh(range(len(num_list)), num_list,tick_label = name_list)
plt.savefig('./picture/'+' '+str(random.randint(0+20*num,19+20*num))+' '+label[p_1]+str(pre_s[0])+' '+label[p_2]+str(pre_s[1])+' '+label[p_3]+str(pre_s[3])+'.jpg')
plt.show()