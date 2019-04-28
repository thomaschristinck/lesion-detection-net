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

# A script to demo everything - unfortunately the data used for this project
# cannot be shared so this is pretty much useless

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
IMAGE_DIR = os.path.join(ROOT_DIR, "images2")

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
index = random.randint(0,3)

# Get index of each image
nb_images = 5
t2_idx = (index // nb_images) * nb_images + 3
netseg_idx = (index // nb_images) * nb_images + 1 
threshed_idx = (index // nb_images) * nb_images + 2
target_idx = (index // nb_images) * nb_images 
unc_idx = (index // nb_images) * nb_images + 4


netseg, opts = nrrd.read(join(IMAGE_DIR, file_names[netseg_idx]))
t2, opts = nrrd.read(join(IMAGE_DIR, file_names[t2_idx]))
target, opts = nrrd.read(join(IMAGE_DIR, file_names[target_idx]))
unc, opts = nrrd.read(join(IMAGE_DIR, file_names[unc_idx]))
threshed, opts = nrrd.read(join(IMAGE_DIR, file_names[threshed_idx]))

print('T2 : ', join(IMAGE_DIR, file_names[t2_idx]))
print('Netseg : ', join(IMAGE_DIR, file_names[netseg_idx]))
print('Uncertainty : ', join(IMAGE_DIR, file_names[unc_idx]))
print('Grount truth : ', join(IMAGE_DIR, file_names[target_idx]))
print('Output : ', join(IMAGE_DIR, file_names[threshed_idx]))
netseg = np.asarray(netseg)
t2 = np.asarray(t2)
target = np.asarray(target)
unc = np.asarray(unc)
threshed = np.asarray(threshed)

threshed, _ = utils.remove_tiny_les(threshed, nvox=2)
target, _ = utils.remove_tiny_les(target, nvox=2)

# Build the 3d image to be viewed
visualize.build_image3d(t2, target, netseg, unc, threshed, model, class_names)




