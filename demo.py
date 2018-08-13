import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from os.path import join

import utils
import model as modellib
import visualize

import random
import nrrd
import launcher
import torch
import utils
from scipy import ndimage
from launcher import MSDataset
from config import Config

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(launcher.DataConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
MODEL_PATH = config.CONTINUE_MODEL_PATH

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Output directory to save roc curves etc.
out_dir = '/usr/local/data/thomasc/logs/'

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config, thresh=0.95)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights 
model.load_state_dict(torch.load(MODEL_PATH))


# Index of the class in the list is its ID.
class_names = ['BG', 'lesion']

# Load a random image from the images folder
file_names = sorted(os.listdir(IMAGE_DIR))
index = random.randint(15,20)
#10, 15 is good


# Get index of each image
bbox_idx = (index // 6) * 6
netseg_idx = (index // 6) * 6 + 1 
t2_idx = (index // 6) * 6 + 2
target_idx = (index // 6) * 6 + 3
unc_idx = (index // 6) * 6 + 4
threshed_idx = (index // 6 ) * 6 + 5

netseg, opts = nrrd.read(join(IMAGE_DIR, file_names[netseg_idx]))
t2, opts = nrrd.read(join(IMAGE_DIR, file_names[t2_idx]))
target, opts = nrrd.read(join(IMAGE_DIR, file_names[target_idx]))
unc, opts = nrrd.read(join(IMAGE_DIR, file_names[unc_idx]))
threshed, opts = nrrd.read(join(IMAGE_DIR, file_names[threshed_idx]))

print('File being viewed : ', join(IMAGE_DIR, file_names[t2_idx]))
netseg = np.asarray(netseg)
t2 = np.asarray(t2)
target = np.asarray(target)
unc = np.asarray(unc)
threshed = np.asarray(threshed)

threshed, _ = utils.remove_tiny_les(threshed, nvox=2)
target, _ = utils.remove_tiny_les(target, nvox=2)
'''

gt_idx = (index // 5) * 5
netseg_idx = (index // 5) * 5 + 1 
out_idx = (index // 5) * 5 + 2
t2_idx = (index // 5) * 5 + 3
unc_idx = (index // 5) * 5 + 4


netseg, opts = nrrd.read(join(IMAGE_DIR, file_names[netseg_idx]))
t2, opts = nrrd.read(join(IMAGE_DIR, file_names[t2_idx]))
target, opts = nrrd.read(join(IMAGE_DIR, file_names[gt_idx]))
unc, opts = nrrd.read(join(IMAGE_DIR, file_names[unc_idx]))
out, opts = nrrd.read(join(IMAGE_DIR, file_names[out_idx]))

print('idx t2 : ', t2_idx)
print('idx netseg : ', netseg_idx)
print('idx gt  : ', gt_idx)
print('idx unc : ', unc_idx)
print('idx out : ', out_idx)

print('File being viewed : ', join(IMAGE_DIR, file_names[t2_idx]))
print('File being viewed : ', join(IMAGE_DIR, file_names[netseg_idx]))
print('File being viewed : ', join(IMAGE_DIR, file_names[gt_idx]))
print('File being viewed : ', join(IMAGE_DIR, file_names[unc_idx]))
print('File being viewed : ', join(IMAGE_DIR, file_names[out_idx]))

netseg = np.asarray(netseg)
t2 = np.asarray(t2)
target = np.asarray(target)
unc = np.asarray(unc)
out = np.asarray(out)
'''

#netseg, _ = utils.remove_tiny_les(netseg, nvox=2)
#target, _ = utils.remove_tiny_les(target, nvox=2)

# Build the 3d image to be viewed
visualize.build_image3d(t2, target, netseg, unc, threshed, model, class_names)




