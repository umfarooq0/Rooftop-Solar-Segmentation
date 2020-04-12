## load keras model and test
from keras.models import load_model
import numpy as np
import glob
import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
import os
from keras import backend as K



img_h = 512
img_w = 512

def iou_score(target,prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score



## load keras model
model_name = 'RooftopSolar_BC.h5'
path = '/home/umfarooq0/RooftopSolar/'
model = load_model(path + model_name)
print('model loaded')
 #where the masks are location
mask_location = path
# where the images are located
test_images_loc = path + 'test_data'

test_images= [f for f in listdir(test_images_loc) if isfile(join(test_images_loc, f))]

iou_result = {}
ii = 0.0
for ti in test_images:
    ii+= 1.00
    ti_ = ti.split('.')[0]
# get images
    image = cv2.imread(test_images_loc + '/' + ti, 0)

    mf = mask_location + 'test_masks' + ti_ + '.txt.npz'

    lc = os.path.exists(mf)
    if lc is True:
    # get masks for that image
        mask = np.load(mf)
        mask_ = np.empty((img_h, img_w, 1))

        mask_resized = cv2.resize(mask.f.arr_0,(img_w,img_h))
        mask_resized = np.array(mask_resized,dtype = np.float64)

        mask_[...,0] = mask_resized

    ## preprocess image
        image_resized = cv2.resize(image, (img_w, img_h))
        image_resized = np.array(image_resized, dtype=np.float64)
        # standardize image

        sd_image = np.zeros((img_h,img_w))
        sd_image = cv2.normalize(image_resized,sd_image,0,255,cv2.NORM_MINMAX)
        X = np.empty((1, img_h, img_w, 1))

        X[0,] = np.expand_dims(sd_image, axis=2)
        
        result = model.predict(X)

        # calculate iou
        accuracy = iou_score(mask_,result)

        iou_result[ti_] = accuracy
        print(accuracy)
    print(ii/len(test_images))
