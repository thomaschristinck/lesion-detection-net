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
from analyze.mrcnn_analyzer import MRCNNAnalyzer as Analyzer


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
config.display()

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
MODEL_PATH = config.CONTINUE_MODEL_PATH

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Output directory to save roc curves etc.
out_dir = '/usr/local/data/thomasc/logs/'

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights 
model.load_state_dict(torch.load(MODEL_PATH))


# Index of the class in the list is its ID.
class_names = ['BG', 'lesion']

# Load a random image from the images folder
file_names = sorted(os.listdir(IMAGE_DIR))
index = random.randint(0,15)
slice_index = random.randint(20, 35)

netseg_idx = (index // 4) * 4 
t2_idx = (index // 4) * 4 + 1
target_idx = (index // 4) * 4 + 2
unc_idx = (index // 4) * 4 + 3

netseg, opts = nrrd.read(join(IMAGE_DIR, file_names[netseg_idx]))
t2, opts = nrrd.read(join(IMAGE_DIR, file_names[t2_idx]))
target, opts = nrrd.read(join(IMAGE_DIR, file_names[target_idx]))
unc, opts = nrrd.read(join(IMAGE_DIR, file_names[unc_idx]))

netseg = np.asarray(netseg)

t2 = np.asarray(t2)

target = np.asarray(target)
unc = np.asarray(unc)

'''
target_slice, _ = utils.remove_tiny_les(target[:,:,slice_index])
netseg_slice = netseg[:,:,slice_index]
t2_slice = t2[:,:,slice_index]
unc_slice = unc[:,:,slice_index]

# Stack slices to make the input image
image_slice = np.stack([t2_slice, unc_slice, netseg_slice], axis = 0)

# Run detection
results = model.detect([image_slice])
image_slice = image_slice.transpose(1,2,0)

# Visualize results
r = results[0]

# Visualize bounding boxes with target lesions
visualize.build_image(image_slice, target_slice, r['rois'], r['masks'], netseg_slice, r['class_ids'], class_names, r['scores'])
'''

# Stack slices to make the input image
#image_slice = np.stack([t2, unc, netseg], axis = 0)

#r = results[0]
#image_slice = image_slice.transpose(1,2,0)

visualize.build_image3d(t2, target, netseg, unc, model, class_names)





'''
dataset_train = MSDataset()
dataset_train.load_data('/usr/local/data/thomasc/unet_out/3d_all_img_small', {'mode': 'train', 'shuffle': True if config.SHUFFLE is 1 else False, 'dim': config.BRAIN_DIMENSIONS,'mods': config.MODALITIES})    

test_set = modellib.Dataset(dataset_train, config, augment=True)
data_gen = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)
full_analyzer = Analyzer(model, MODEL_PATH, data_gen, join(out_dir, 'cca'), nb_mc=10)
full_analyzer.roc(join(out_dir, 'cca'), thresh_start=0, thresh_stop=1, thresh_step=0.05)
'''