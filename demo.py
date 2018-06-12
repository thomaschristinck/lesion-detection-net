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


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_lesion_mask_rcnn_0016.pth")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(launcher.DataConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(MODEL_PATH))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'lesion']

# Load a random image from the images folder
file_names = sorted(os.listdir(IMAGE_DIR))
index = random.randint(0,23)
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

image = np.stack([t2[:,:,slice_index], unc[:,:,slice_index], netseg[:,:,slice_index]], axis = 0)

# Run detection
results = model.detect([image])

image = image.transpose(1,2,0)
netseg = netseg[:,:,slice_index]
target = target[:,:,slice_index]

# Visualize results
r = results[0]

# Visualize bounding boxes with target lesions
print('Plotting stuff.....')
visualize.display_instances(image, target, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

# View the 'ground truth' segmentation
plt.figure(1)
plt.imshow(t2[:,:,slice_index], interpolation='nearest', cmap=plt.cm.pink)
plt.axis('off')
plt.suptitle('Network Bounding Boxes with Ground Truth Segmentation')

# View the segmentation output of the BUnet
plt.figure(2)
plt.imshow(netseg, cmap=plt.cm.pink)
plt.axis('off')
plt.suptitle('BUnet Segmentation')

# View T2
plt.figure(3)
plt.imshow(t2[:,:,slice_index], cmap=plt.cm.pink)
plt.axis('off')
plt.suptitle('T2 Image')

plt.show()