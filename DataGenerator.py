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



class DataGenerator(Sequence):
    def __init__(self, list_ids, labels, image_dir, batch_size=5,
                 img_h=512, img_w=512, shuffle= False):
        #self.steps_per_epoch = steps_per_epoch
        self.list_ids = list_ids
        self.labels = labels
        self.image_dir = image_dir
        self.batch_size = batch_size

        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids)) / self.batch_size)

    def __getitem__(self, index):
        'generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(list_ids_temp)
        # return data
        return X, y

    def on_epoch_end(self):
        'update ended after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        'generate data containing batch_size samples'
        X = np.empty((self.batch_size, self.img_h, self.img_w, 1))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1)) #  this was originally 4, but changed to 1


        for idx, id in enumerate(list_ids_temp):
            file_path =  os.path.join(self.image_dir, id+'.tif')
            lc = os.path.exists(file_path)
            if lc is True:

                image = cv2.imread(file_path, 0)
                im_sz = image.size

                if im_sz > 0:

                    image = image/255.0


                    image_resized = cv2.resize(image, (self.img_w, self.img_h))
                    image_resized = np.array(image_resized, dtype=np.float64)

                    mask = np.empty((self.img_h, self.img_w, 1))

                    rle = self.labels.get(id)
                    rle = rle.f.arr_0
                    if rle is None:
                        class_mask = np.zeros((5000, 5000))
                    else:
                        class_mask = rle

                    class_mask_resized = cv2.resize(class_mask, (self.img_w, self.img_h))
                    mask[...,0] = class_mask_resized

                    X[idx,] = np.expand_dims(image_resized, axis=2)
                    y[idx,] = mask


        # normalize Y
        y = (y > 0).astype(int)

        return X,y
