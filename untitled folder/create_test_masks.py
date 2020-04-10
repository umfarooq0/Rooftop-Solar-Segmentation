## this file will create the test data set to be used

from os import listdir
from os.path import isfile, join
from create_masks import create_masks
import numpy as np

import pandas as pd

## we need to load each file from test data set and create the mask accordingly

path = '/home/umfarooq0/RooftopSolar/'

test_data_loc = path + 'test_data'
## import dataset
df = pd.read_csv(path + 'sample_data.csv')

## files in test data folder
img_h = 5000
img_w = 5000

# give list of filenames
testfiles = [f for f in listdir(test_data_loc) if isfile(join(test_data_loc, f))]
location = 'test_masks'
# load files
i = 0.0
for fn in testfiles:
    i += 1.0
    filename = fn.split('.')[0]
    mask = create_masks(filename,df,img_h,img_w,location)
    print(i/len(testfiles))
