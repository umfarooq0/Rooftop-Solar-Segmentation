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

from matplotlib.path import Path

import json
import time
import pickle
from create_masks import create_masks
from DataGenerator import DataGenerator
from models import unet


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

train_image_ids = in_un
val_size = 20
train_image_ids = train_image_ids[train_image_ids.image_name != '11ska505815']
train_image_ids = train_image_ids[train_image_ids.image_name != '10sfh465105']

X_train, X_val = train_test_split(train_image_ids, test_size=val_size, random_state=42)

tfnames = sample_data['image_name'].unique()

#Split tfnames into 4 parts and run each part

first,second,third,fourth = np.array_split(tfnames,4)



print('STARTING MASK GENERATION')

'''
img_masks1 = [create_masks(x,sample_data,5000,5000) for x in first]
tmasks1 = dict(zip(first,img_masks1))
print('COMPLETED MASK1')
'''
img_masks1 = pickle.load( open( "image_masks1.pickle", "rb" ) )
img_masks2_1 = pickle.load( open( "image_masks2_1.pickle", "rb" ) )
print('LOADED MASKS 1 AND 2_1')
#print('STARTING MASK2_1')
second_one,second_two,second_three = np.array_split(second,3)

'''

img_masks2_1 = [create_masks(x,sample_data,5000,5000) for x in second_one]

tmasks2_1 = dict(zip(second_one,img_masks2_1))


print(end_pickle)
print('COMPLETED MASK2_1')
'''

print('STARTING MASK2_2')
img_masks2_2 = [create_masks(x,sample_data,5000,5000) for x in second_two]
tmasks2_2 = dict(zip(second_two,img_masks2_2))

print('COMPLETED MASK2_2')

print('STARTING MASK2_3')
img_masks2_3= [create_masks(x,sample_data,5000,5000) for x in second_three]
tmasks2_3 = dict(zip(second_three,img_masks2_3))


print('COMPLETED MASK2_3')

print('STARTING MASK 3')
img_masks3 = [create_masks(x,sample_data,5000,5000) for x in third]
tmasks3 = dict(zip(third,img_masks3))
#np.savetxt('img_masks3.txt',img_masks3)

print('COMPLETED MASK3')

print('STARTING MASK 4')
img_masks4 = [create_masks(x,sample_data,5000,5000) for x in fourth]
tmasks4 = dict(zip(fourth,img_masks4))

print('COMPLETED MASK4')

print('FINISHED MASK GENERATION')

complete_masks = {}
all_m = (img_masks1,img_masks2_1,tmasks2_2,tmasks2_3,tmasks3,tmasks4)
for d in all_m:
    complete_masks.update(d)

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

training_generator = DataGenerator(X_train, complete_masks, **params)
validation_generator = DataGenerator(X_val, complete_masks, **params)

model = unet()
epochs = 25


history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, verbose=1)

model.save('RooftopSolar_BC.h5')
