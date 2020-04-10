import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy import sparse

path = '/home/umfarooq0/RooftopSolar/'

def create_masks(img_name, df,img_h,img_w,loc):
    path = '/home/umfarooq0/RooftopSolar/'
        # input id/image name
            # df is the 'sample data frame', houses all the image names and the corresponding RLE's
    data = df[df['image_name'] == img_name]
                    # this will give us all the RLE's related that image name
    rle_size  = len(data)
                            # we need to loop through each set of RLE and create a list of all the points
    RLE = data['join'].reset_index(drop = True)

    x, y = np.meshgrid(np.arange(img_h), np.arange(img_w)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    if rle_size == 0:
        return

    for i in range(rle_size):
        print(i)
        
        RLE_string = RLE[i]
        rlen = [int(float(numstring)) for numstring in RLE_string.split(' ')]
        rlePairs = np.array(rlen).reshape(-1,2)
        p = Path(rlePairs) # make a polygon
        grid = p.contains_points(points)
        if i == 0:
            grid_ = np.zeros((img_h*img_w))
            grid_ = grid + grid_
        else:
            grid_ += grid
    
    mask = grid_.reshape(img_h,img_w)

    
    filename_save = path + loc + img_name + '.txt'
    np.savez_compressed(filename_save, mask)

    return mask
