#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
import cv2
from keras.utils import Sequence
from scipy import sparse

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping
from matplotlib.path import Path

import json
import time
import pickle
from create_masks import create_masks
from DataGenerator import DataGenerator
from models import unet

from keras.callbacks import  ModelCheckpoint

from os import listdir
from os.path import isfile,join



## Import required data and join


path = '/home/umfarooq0/RooftopSolar/'

long_lat_file = 'polygonVertices_LatitudeLongitude.csv'

pol_long_lat = pd.read_csv(path + long_lat_file)

pol_long_lat.head()


pixel_coord_file = 'polygonVertices_PixelCoordinates.csv'

pol_coord = pd.read_csv(path + pixel_coord_file)

pol_coord.shape

pol_coord.iloc[0,:]



pol_coord['join'] = pol_coord[pol_coord.columns[2:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1)

except_vert_file = 'polygonDataExceptVertices.csv'

except_vert = pd.read_csv(path + except_vert_file)


def create_class(x):
    if x['polygon_id'] < 20:
        return int(0)
    elif x['polygon_id'] == 20:
        return int(1)
    elif 20 < x['polygon_id'] < 50:
        return int(2)
    elif x['polygon_id'] > 50:
        return int(3)

df2_vals = except_vert.groupby(['image_name']).count()['polygon_id']
df2_vals = pd.DataFrame(df2_vals)
df2_vals['class'] = df2_vals.apply(create_class,axis = 1)

df_coord = pd.merge(except_vert,pol_coord,on = 'polygon_id')

in_un = except_vert.image_name.unique()

len(except_vert.image_name.unique())

in_un = pd.DataFrame(in_un,columns = ['image_name'])

sample_data = in_un.merge(df_coord,how = 'inner', on='image_name')

sample_data['join'] = sample_data['join'].apply(lambda x: x.replace(","," "))

sample_data.to_csv('sample_data.csv')
train_image_dir = path + 'train_data'


train_image_ids =  [f for f in listdir(train_image_dir) if isfile(join(train_image_dir, f))]


val_size = 20
'''
train_image_ids = train_image_ids[train_image_ids.image_name != '11ska505815']
train_image_ids = train_image_ids[train_image_ids.image_name != '10sfh465105']
'''

X_train, X_val = train_test_split(train_image_ids, test_size=val_size, random_state=42)

tfnames = sample_data['image_name'].unique()

#Split tfnames into 4 parts and run each part



import sys

print('STARTING MASK GENERATION')

'''
img_masks1 = [create_masks(x,sample_data,5000,5000) for x in first]
tmasks1 = dict(zip(first,img_masks1))
print('COMPLETED MASK1')
'''

'''
img_masks1 = pickle.load( open( "masks1.pickle", "rb" ) )
img_masks2_1 = pickle.load( open( "masks2_1.pickle", "rb" ) )
m1 = sys.getsizeof(img_masks1)
print(m1)
m2 = sys.getsizeof(img_masks2_1)
print(m2)

print('LOADED MASKS 1 AND 2_1')
#print('STARTING MASK2_1')
second_one,second_two,second_three = np.array_split(second,3)
'''
'''

img_masks2_1 = [create_masks(x,sample_data,5000,5000) for x in second_one]

tmasks2_1 = dict(zip(second_one,img_masks2_1))


print(end_pickle)
print('COMPLETED MASK2_1')
'''
'''
print('STARTING MASK2_2')
img_masks2_2 = [create_masks(x,sample_data,5000,5000) for x in second_two]
tmasks2_2 = dict(zip(second_two,img_masks2_2))

print('COMPLETED MASK2_2')

print('STARTING MASK2_3')
img_masks2_3= [create_masks(x,sample_data,5000,5000) for x in second_three]
tmasks2_3 = dict(zip(second_three,img_masks2_3))


print('COMPLETED MASK2_3')
'''
'''
print('STARTING MASK 3')
third_one,third_two,third_three = np.array_split(third,3) 

img_masks3_1 = [create_masks(x,sample_data,5000,5000) for x in third_one]
tmasks3_1 = dict(zip(third_one,img_masks3_1))
print(' mask3_1 complete')

img_masks3_2 = [create_masks(x,sample_data,5000,5000) for x in third_two]
tmasks3_2 = dict(zip(third_two,img_masks3_2))
print('mask3_2 complete')

img_masks3_3 = [create_masks(x,sample_data,5000,5000) for x in third_three]
tmasks3_3 = dict(zip(third_three,img_masks3_3))
print('COMPLETED MASK3_3')
'''
'''
print('STARTING MASK 4')
img_masks4 = [create_masks(x,sample_data,5000,5000) for x in fourth]
tmasks4 = dict(zip(fourth,img_masks4))

print('COMPLETED MASK4')
'''
print('FINISHED MASK GENERATION')
'''
complete_masks = {}
all_m = (img_masks1,img_masks2_1,tmasks2_2,tmasks2_3,tmasks3,tmasks4)
for d in all_m:
    complete_masks.update(d)
'''

mask_path = path + 'masks'


maskfiles = [f for f in listdir(mask_path) if isfile(join(mask_path, f))]

mask_data = {}
i = 0.0
for fn in maskfiles:
    i += 1.0
    fl = np.load(mask_path +'/' + fn)
    mn = fn.split('.')[0]
    print(i/len(maskfiles))
    mask_data[mn] = fl


img_h = 512
img_w = 512
train_image_dir = path + 'train_data'
batch_size = 5

params = {'img_h': img_h,
          'img_w': img_w,
          'image_dir': train_image_dir,
          'batch_size': batch_size,
          'shuffle': True}

X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0])

X_val = np.array(X_val)
X_val = X_val.reshape(X_val.shape[0])

training_generator = DataGenerator(X_train, mask_data, **params)
validation_generator = DataGenerator(X_val, mask_data, **params)

import segmentation_models as sm

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)



rooftopsolar_model1 = sm.Unet(BACKBONE,encoder_weights = 'imagenet')
rooftopsolar_model1.compile('Adam',loss = sm.losses.bce_jaccard_loss,metrics = [sm.metrics.iou_score],)

epochs = 80


early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 5)
checkpointer = ModelCheckpoint('rooftopsolar_model6.h5',monitor = 'loss', verbose=1, save_best_only=True)


rooftopsolar_model1.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs,callbacks = [early_stopping_callback,checkpointer], verbose=1)


