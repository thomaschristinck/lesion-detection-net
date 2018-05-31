
import os
import time
import numpy as np

import zipfile
import urllib.request
import shutil
import random

from config import Config
import utils
from os.path import join
from os import listdir
import model as modellib
from model import Dataset

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "lesion_mask_rcnn.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class DataConfig(Config):
	"""Configuration for training on MSLAQ data.
	Derives from the base Config class and overrides values specific to the way I've set up
	the hdf5 files
	"""
	# Give the configuration a recognizable name
	NAME = "mask_hdf5"

	# We use one GPU with 12GB memory (I think), which can fit ~one image.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 16

	# Uncomment to train on 8 GPUs (default is 1)
	# GPU_COUNT = 8

	# Number of classes (including background). Small, medium, and large lesions (each is a 'class')
	NUM_CLASSES = 4


class MSDataset(utils.Dataset):

	def load_data(self, dataset_dir, config, class_ids=None,
					class_map=None, return_object=False):
		""" TODO: Remove this and fix model.Dataset

		Load a subset of the dataset. TODO: Remove this and fix model.Dataset
		dataset_dir: The root directory of the dataset.
		subset: What to load (train, val)
		class_ids: TODO: If provided, only loads images that have the given classes.
		class_map: TODO: Not implemented yet. Supports maping classes from
			different datasets to the same class ID.
		return_object: TODO: If True, returns the object.
		"""

		self._image_list = sorted(listdir(dataset_dir))
		self._dir = dataset_dir
		self._mode = config.get('mode')
		self._shuffle = config.get('shuffle', True)
		self._image_ids = np.asarray(self._image_list)
		#self._image_ids = [i[4:] for i in self._image_ids]
		self._nb_folds = config.get('nb-folds', 10)
		self._config = config

		if config.get('dim') == 2:
			#Going to just try looking at a random slice for now
			slice_index = random.randint(0, len(config.get('mods')))
			self._slice_index = slice_index
		else:
			slice_idx = ...  
			self._slice_index = slice_index


		print('Slice index : ', self._slice_index)

		fold_length = len(self._image_ids) // self._nb_folds
		#self._image_ids = self._rotate(self._image_ids, config.get('fold', 0) * fold_length)
		train_idx = (((self._nb_folds - 2) * fold_length) // 7) * 7
		valid_idx = (((self._nb_folds - 1) * fold_length) // 7) * 7
		
		if self._mode == 'train':
			self._image_ids= self._image_ids[:train_idx]
		elif self._mode == 'val':
			self._image_ids = self._image_ids[train_idx:valid_idx]
		elif self._mode == 'test':
			self._image_ids = self._image_ids[valid_idx:]


		class_ids = ['small', 'medium', 'large']
		for idx, size in enumerate(class_ids):
			idx += 1
			self.add_class("MSLAQ", idx, size)
		for i in self._image_ids:
			self.add_image("MSLAQ", image_id=i, path = os.path.join(dataset_dir, i))

	def load_mask(self, image_id):
		"""TODO: Remove this function

		Load instance masks for the given image.

		Given image should have lesion masks. This function converts the mask format to 
		format the form of a bitmap [height, width, instances].

		Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		# If not a COCO image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "coco":
			return super(CocoDataset, self).load_mask(image_id)

		instance_masks = []
		class_ids = []
		annotations = self.image_info[image_id]["annotations"]
		# Build mask of shape [height, width, instance_count] and list
		# of class IDs that correspond to each channel of the mask.
		for annotation in annotations:
			class_id = self.map_source_class_id(
				"coco.{}".format(annotation['category_id']))
			if class_id:
				m = self.annToMask(annotation, image_info["height"],
								   image_info["width"])
				# Some objects are so small that they're less than 1 pixel area
				# and end up rounded out. Skip those objects.
				if m.max() < 1:
					continue
				# Is it a crowd? If so, use a negative class ID.
				if annotation['iscrowd']:
					# Use negative class ID for crowds
					class_id *= -1
					# For crowd masks, annToMask() sometimes returns a mask
					# smaller than the given dimensions. If so, resize it.
					if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
						m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
				instance_masks.append(m)
				class_ids.append(class_id)

		# Pack instance masks into an array
		if class_ids:
			mask = np.stack(instance_masks, axis=2)
			class_ids = np.array(class_ids, dtype=np.int32)
			return mask, class_ids
		else:
			# Call super class to return an empty mask
			return super(CocoDataset, self).load_mask(image_id)

	# The following two functions are from pycocotools with a few changes.

	def annToRLE(self, ann, height, width):
		"""
		Convert annotation which can be polygons, uncompressed RLE to RLE.
		:return: binary mask (numpy 2D array)
		"""
		segm = ann['segmentation']
		if isinstance(segm, list):
			# polygon -- a single object might consist of multiple parts
			# we merge all parts into one mask rle code
			rles = maskUtils.frPyObjects(segm, height, width)
			rle = maskUtils.merge(rles)
		elif isinstance(segm['counts'], list):
			# uncompressed RLE
			rle = maskUtils.frPyObjects(segm, height, width)
		else:
			# rle
			rle = ann['segmentation']
		return rle

	def annToMask(self, ann, height, width):
		"""
		Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
		:return: binary mask (numpy 2D array)
		"""
		rle = self.annToRLE(ann, height, width)
		m = maskUtils.decode(rle)
		return m

	@staticmethod
	def _rotate(l, n):
		return l[-n:] + l[:-n]

if __name__ == '__main__':
	import argparse

	config = DataConfig()
	
	config.display()

	# Create model
 
	model = modellib.MaskRCNN(config=config,
								  model_dir='usr/local/data/thomasc')
	if config.GPU_COUNT:
		model = model.cuda()

	# Select weights file to load
 

	# Training and validation datasets. Later: for training use the training set and 35K from the
	# validation set, as as in the Mask RCNN paper.
	'''
	train_ds = DataProvider(expt_cfg['data_path'],
							{'mode': 'train', 'shuffle': True if expt_cfg['shuffle'] is 1 else False})
	valid_ds = DataProvider(expt_cfg['data_path'], {'mode': 'valid', 'shuffle': False})
	train_gen = train_ds.get_generator(expt_cfg['batch_size'], expt_cfg['nb_epochs'])
	valid_gen = valid_ds.get_generator(expt_cfg['batch_size'], expt_cfg['nb_epochs'])
	
	train_set = Dataset(train_dataset, self.config, augment=True)
	val_set = Dataset(val_dataset, self.config, augment=True)
	'''

   
	# Train or evaluate
		# Training dataset (possibly modify so some examples come from validation set as in MaskRCNN paper)


	dataset_train = MSDataset()
	dataset = '/usr/local/data/thomasc/unet_out/all_img'
	dataset_train.load_data(dataset, {'mode': 'train', 'shuffle': True if config.SHUFFLE is 1 else False, 'dim': config.BRAIN_DIMENSIONS,'mods': config.MODALITIES})    
	dataset_train.prepare()

		# Validation dataset
	dataset_val = MSDataset()
	dataset_val.load_data(dataset, {'mode': 'val', 'shuffle': False, 'dim': config.BRAIN_DIMENSIONS, 'mods': config.MODALITIES})
	dataset_val.prepare()

	train_set = Dataset(dataset_train, config, augment=True)
	val_set = Dataset(dataset_val, config, augment=True)

	train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
	val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4)
	
	print(val_generator)
	for inputs in val_generator:
		batch_count += 1
		print('Inputs: ', inputs)
		images = inputs[0]
		image_metas = inputs[1]
		rpn_match = inputs[2]
		rpn_bbox = inputs[3]
		gt_class_ids = inputs[4]
		gt_boxes = inputs[5]
		gt_masks = inputs[6]
		print(images.shape, images.size)