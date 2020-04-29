## take each image and make it into blocks
import numpy as np
import cv2
import shutil
import os
 ## get list of images

 mypath  = '/home/umfarooq0/RooftopSolar/train_data/'
 os.chdir(mypath)
 # get all files in folder
 onlyfiles = [f for f in listdir(mypath) if if f.endswith('.tif')]


 for x in onlyfiles:
  # load image
  img_loc = mypath + x
  img = cv2.imread(img_loc)
  ## no of pieces you want to split the image into
  pieces = 10

  M = img.shape[0]//pieces
  N = img.shape[1]//pieces

  tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]

  for i in range(len(tiles)):
    fn = mypath + 'train_split_images/' + x + '_' + i
    np.savez_compressed(fn,tiles[i])

  os.remove(x)


11ska490665.tif
11ska580830.tif
11ska505830.tif
11ska445860.tif
11ska580695.tif

/home/umfarooq0/RooftopSolar/run
