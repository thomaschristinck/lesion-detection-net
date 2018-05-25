from os.path import join
import numpy as np
from scipy import ndimage
import h5py
import matplotlib
import matplotlib.pyplot as plt
import visualize
from box_generator import bounding_boxes


in_dir = '/usr/local/data/thomasc/det_in'
f = h5py.File(join(in_dir, 'det_data.h5py'))
dataset_target = f['vmk_9581505']['m0']['target']
dataset_t2 = f['vmk_9581505']['m0']['t2']
mask = np.asarray(dataset_target)
t2 = np.asarray(dataset_t2)
dict = bounding_boxes(mask, t2, sm_buf=1, med_buf=3, lar_buf=5)
print(dict['classes'])
print(dict['boxes'])
