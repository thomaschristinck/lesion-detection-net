"""
Mask R-CNN
Configurations and data loading code for the MSLAQ dataset. Heavily modified 
code originally written by Waleed Abdulla for the COCO dataset.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: run from the command line as such:

	# Continue training a model that you had trained earlier
	python3 data_loader.py train --dataset=/path/to/dataset/ --model=/path/to/weights.h5

	# Continue training the last model you trained
	python3 data_loader.py train --dataset=/path/to/dataset/ --model=last

	# Run evaluatoin on the last model you trained
	python3 data_loader.py evaluate --dataset=/path/to/dataset/ --model=last
"""

import os
import time
import numpy as np
import random

from config import Config
import utils
from os.path import join
from os import listdir
import model as modellib
from model import Dataset
from scipy import ndimage

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "lesion_mask_rcnn.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

###############################################################
# Data set
###############################################################

class DataConfig(Config):
	"""Configuration for training on MSLAQ data.
	Derives from the base Config class and overrides values specific to the way I've set up
	the hdf5 files
	"""
	# Give the configuration a recognizable name
	NAME = "lesion_mask_rcnn"

	# We use one GPU with 12GB memory (I think), which can fit ~one image.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 16

	# Uncomment to train on more GPUs (default is 1)
	# GPU_COUNT = 8

	# Number of classes (including background). Small, medium, and large lesions (each is a 'class')
	NUM_CLASSES = 2


class MSDataset(utils.Dataset):

	def load_data(self, dataset_dir, config):
		""" 
		Load a subset of the dataset.
		dataset_dir: The root directory of the dataset.
		"""

		self._image_list = sorted(listdir(dataset_dir))
		self._dir = dataset_dir
		self._mode = config.get('mode')
		self._shuffle = config.get('shuffle', True)
		self._image_ids = np.asarray(self._image_list)
		self._nb_folds = config.get('nb-folds', 10)
		self._config = config
		self._slice_idx = 0
		if config.get('dim') == 2:
			#Going to just try looking at a random slice for now
			slice_idx = random.randint(0,63)
			self._slice_idx = slice_idx
		else:
			slice_idx = ...  
			self._slice_idx = slice_idx

		fold_length = len(self._image_ids) // self._nb_folds
		mod_number = len(config.get('mods'))
		train_idx = (((self._nb_folds - 2) * fold_length) // mod_number) * mod_number
		valid_idx = (((self._nb_folds - 1) * fold_length) // mod_number) * mod_number
		
		if self._mode == 'train':
			self._image_ids= self._image_ids[:train_idx]
		elif self._mode == 'val':
			self._image_ids = self._image_ids[train_idx:valid_idx]
		elif self._mode == 'test':
			self._image_ids = self._image_ids[valid_idx:valid_idx + (7 * 25)]

		# Build (or rebuild) everything else from the info dicts.
		self.num_images = int(len(self._image_ids) / 8)
		self._image_ids = np.arange(self.num_images)
		
	@staticmethod
	def _rotate(l, n):
		return l[-n:] + l[:-n]


###################################################################
#  Training - Possibly will start with Imagenet pretrained weights
###################################################################

if __name__ == '__main__':
	import argparse

	  # Parse command line arguments
	parser = argparse.ArgumentParser(description='Train Mask R-CNN on MSLAQ dataset.')
	parser.add_argument("command",
						metavar="<command>",
						help="'train' or 'evaluate'")
	parser.add_argument('--dataset', required=True,
						metavar="/path/to/mslaq.h5",
						help='Directory of the dataset')
	parser.add_argument('--model', required=False,
						metavar="-m /path/to/weights.pth",
						help="Path to weights .pth file")
	parser.add_argument('--logs', required=False,
						default=DEFAULT_LOGS_DIR,
						metavar="-l /path/to/logs/",
						help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--limit', required=False,
						default=50,
						metavar="<image count>",
						help='Images to use for evaluation (default=50)')
	
	args = parser.parse_args()
	print("Command: ", args.command)
	print("Model: ", args.model)
	print("Dataset: ", args.dataset)

	# Configurations
	if args.command == 'train':
		config = DataConfig()
	else:
		class InferenceConfig(DataConfig):
			# Set batch size to 1
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
			DETECTION_MIN_CONFIDENCE = 0
		config = InferenceConfig()	
	config.display()

	# Create model
 
	model = modellib.MaskRCNN(config=config, model_dir=args.logs, thresh=0)
	if config.GPU_COUNT:
		model = model.cuda()

	# Select weights file to load
	if args.model:
		if args.model.lower() == "continue":
			# Start from weights (path set in config)
			model_path = config.CONTINUE_MODEL_PATH
		elif args.model.lower() == "imagenet":
			# Start from ImageNet trained weights
			model_path = config.IMAGENET_MODEL_PATH
		else:
			#Start from weights (path specified as --model=/path/to/weights)
			model_path = args.model
	else:
		model_path = ""

	
	# Load weights
	print("Loading weights ", model_path)
	model.load_weights(model_path)
 
	# Train or evaluate
	if args.command == "train":

		# Training dataset (possibly modify so some examples come from validation set as in MaskRCNN paper)
		dataset_train = MSDataset()
		dataset_train.load_data(args.dataset, {'mode': 'train', 'shuffle': True if config.SHUFFLE is 1 else False, 'dim': config.BRAIN_DIMENSIONS,'mods': config.MODALITIES})    

		# Validation dataset
		dataset_val = MSDataset()
		dataset_val.load_data(args.dataset, {'mode': 'val', 'shuffle': False, 'dim': config.BRAIN_DIMENSIONS, 'mods': config.MODALITIES})

		# Training - Stage 1
		print("Training network heads")
		model.train_model(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE / 100,
					epochs=30,
					layers='heads')

		# Training - Stage 2
		print("Fine tune Resnet stage 4 and up")
		model.train_model(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE / 100,
					epochs=60,
					layers='4+')

		# Training - Stage 3
		print("Fine tune all layers")
		model.train_model(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE / 1000, #Changed from /10
					epochs=85,
					layers='all')
		
		# Training - Stage 4
		print("Train Network heads")
		model.train_model(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE / 1000, #Changed from /10
					epochs=110,
					layers='heads')

		# Training - Stage 5
		print("Fine tune all layers")
		model.train_model(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE / 1000, #Changed from /10
					epochs=140,
					layers='all')


	elif args.command == "evaluate":
		# Test dataset
		dataset_test = MSDataset()
		dataset_test.load_data(args.dataset, {'mode': 'test', 'shuffle': False, 'mods': config.MODALITIES})
	
		print("Running evaluation on {} images.".format(args.limit))
		
		# Evaluate the model (produce TPR/FPR graphs)
		model.evaluate_model_segmentation_by_slice(dataset_test, args.logs)
		model.evaluate_model_detection(dataset_test, args.logs)
		model.evaluate_model_segmentation_holistic(dataset_test, args.logs)
		model.evaluate_model_detection_holistic(dataset_test)

	
		
	else:
		print("'{}' is not recognized. "
			  "Use 'train' or 'evaluate'".format(args.command))