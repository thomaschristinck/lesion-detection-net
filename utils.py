
import sys
import os
import math
import random
import numpy as np
import scipy.misc
import skimage.color
import skimage.io
import skimage
from skimage import transform
import torch
from torch import nn
import visualize
import nrrd
from os.path import join
from scipy import ndimage


import matplotlib.pyplot as plt

############################################################
#  Bounding Boxes
############################################################
'''
Lesion bin and tiny lesion removal functions provided from Tanya's BUnet code
'''

def get_3D_lesion_bin(nvox):
	# Lesion bin - 0 for small lesions, 1 for medium, 2 for large
	if 3 <= nvox <= 10:
		return 1
	elif 11 <= nvox <= 50:
		return 2
	elif nvox >= 51:
		return 3
	else:
		return 1

def get_lesion_bin(nvox):
	# Lesion bin - 0 for small lesions, 1 for medium, 2 for large
	if 3 <= nvox <= 10:
		return 'small'
	elif 11 <= nvox <= 50:
		return 'med'
	elif nvox >= 51:
		return 'large'
	else:
		return 'small'

def get_box_lesion_bin(nvox):
	# Lesion bin - 0 for small lesions, 1 for medium, 2 for large
	if 3 <= nvox <= 100:
		return 'small'
	elif 100 < nvox <= 225:
		return 'med'
	elif nvox > 225:
		return 'large'
	else:
		return 'small'


def get_box_lesion_bin_gen(nvox):
	# Lesion bin - 0 for small lesions, 1 for medium, 2 for large
	if 3 <= nvox <= 135:
		return 'small'
	elif 135 < nvox <= 300:
		return 'med'
	elif nvox > 300:
		return 'large'
	else:
		return 'small'


def remove_tiny_les(lesion_image, nvox=2):
	labels, nles = ndimage.label(lesion_image)
	class_ids = np.zeros([nles, 1], dtype=np.int32)

	for i in range(1, nles + 1):
		nb_vox = np.sum(lesion_image[labels == i])
		if nb_vox <= nvox:
			lesion_image[labels == i] = 0
		
		if nb_vox > 0:
			# Classify as lesion. There is a bug here where if we set lesions less than two voxels big
			# to the background class (nb_vox > nvox), then we crash
			class_ids[i-1] = 1

	class_ids = np.asarray(class_ids)

	if class_ids.size == 0:
		class_ids = np.zeros([1, 1], dtype=np.int32)
		class_ids[0] = 0

	return lesion_image, class_ids

def extract_bboxes(mask, dims, buf):
	"""Compute bounding boxes from masks. Could vary 'buf' based on lesion bin
	classes. Removed this feature for now as the multiple classes confuses the RPN.

	mask: [height, width, slice]. Mask pixels are either 1 or 0.

	Returns: if 2D - bbox array [num_instances, (y1, x1, y2, x2, class_id)].
			if 3D - bbox array [num_instances, (y1, x1, y2, x2, class_id)].

	"""

	labels = {}
	nles = {}
	labels, nles = ndimage.label(mask)
	boxes = np.zeros([nles, 6], dtype=np.int32)
	nb_lesions = nles

	for i in range(1, nles + 1):
		
		mask[labels != i] = 0
		mask[labels == i] = 1
 
		# Now we classify the lesion and apply a buffer based on the lesion class (CHANGE LATER??)
		lesion_size = np.sum(mask[labels == i])

		x_indicies = np.where(np.any(mask, axis=0))[0]
		y_indicies = np.where(np.any(mask, axis=1))[0]
		z_indicies =[]
		for lesion_slice in range(mask.shape[-1]):
			if np.any(mask[...,lesion_slice]):
				z_indicies.append(lesion_slice)
		z_indicies = np.asarray(z_indicies)
   
		if x_indicies.shape[0]:
			x1, x2 = x_indicies[[0, -1]]
			y1, y2 = y_indicies[[0, -1]]
			z1, z2 = z_indicies[[0, -1]]
			x2 += 1
			y2 += 1
			z2 += 1
		
			x1 -= buf; x2 += buf; y1 -= buf; y2 += buf; z1 -= buf; z2 += buf

		else:
			# No mask for this instance
			print('Error - no mask here')
			x1, x2, y1, y2, z1, z2 = 0, 0, 0, 0, 0, 0
	   
	   # We want 1 box (and 1 lesion mask, and one class) for each lesion
		boxes[i - 1] = np.array([y1, x1, y2, x2, z1, z2])
	
	if boxes.size == 0:
		x1, x2, y1, y2, z1, z2 = 0, 0, 0, 0, 0, 0
		boxes = np.zeros([1, 6], dtype=np.int32)
		boxes[0] = np.array([y1, x1, y2, x2, z1, z2])	
	
	# Reset ground truth mask and then we can draw boxes
	for i in range(1, nb_lesions + 1):
		mask[labels == i] = 1

	if dims == 2:
		boxes = np.delete(boxes, 4, 1)
		boxes = np.delete(boxes, 4, 1)
		return boxes.astype(np.int32)
	else:
		return boxes.astype(np.int32)

def compute_simple_iou(box, gt_box):
	"""Calculates IoU of the given box with the array of the given boxes.
	box: 1D vector [y1, x1, y2, x2]
	boxes: [boxes_count, (y1, x1, y2, x2)]

	Note: the areas are passed in rather than calculated here for
		  efficency. Calculate once in the caller to avoid duplicate work.
	"""
	# Calculate intersection areas
	y1 = np.maximum(box[0], gt_box[:, 0])
	y2 = np.minimum(box[2], gt_box[:, 2])
	x1 = np.maximum(box[1], gt_box[:, 1])
	x2 = np.minimum(box[3], gt_box[:, 3])
	intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

	# Compute predicted and ground truth box areas
	box_area = (box[2] - box[0]) * (box[3] - box[1])
	gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

	# Compute and return iou (where the union is the sum of the two boxes areas minus their
	# intersection)
	iou = intersection / float(box_area + gt_box_area - intersection)

	return iou

def get_area(box):
	# Compute area
	box_area = (box[2] - box[0]) * (box[3] - box[1])
	return box_area


def compute_2D_iou(box, boxes, box_area, boxes_area):
	"""Calculates IoU of the given box with the array of the given boxes.
	box: 1D vector [y1, x1, y2, x2, class_id]
	boxes: [boxes_count, (y1, x1, y2, x2, class_id)]
	box_area: float. the area of 'box'
	boxes_area: array of length boxes_count.

	Note: the areas are passed in rather than calculated here for
		  efficency. Calculate once in the caller to avoid duplicate work.
	"""
	# Calculate intersection areas
	y1 = np.maximum(box[0], boxes[:, 0])
	y2 = np.minimum(box[2], boxes[:, 2])
	x1 = np.maximum(box[1], boxes[:, 1])
	x2 = np.minimum(box[3], boxes[:, 3])
	intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
	union = box_area + boxes_area[:] - intersection[:]
	iou = intersection / union
	return iou


def compute_2D_overlaps(boxes1, boxes2):
	"""Computes IoU overlaps between two sets of 2D boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2, class_id)].

	For better performance, pass the largest set first and the smaller second.
	"""
	# Areas of anchors and GT boxes
	area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
	area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

	# Compute overlaps to generate matrix [boxes1 count, boxes2 count]
	# Each cell contains the IoU value.
	overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
	for i in range(overlaps.shape[1]):
		box2 = boxes2[i]
		overlaps[:, i] = compute_2D_iou(box2, boxes1, area2[i], area1)
	return overlaps

def box_2D_refinement(box, gt_box):
	"""Compute refinement needed to transform box to gt_box.
	box and gt_box are [N, (y1, x1, y2, x2, class_id)]
	"""

	height = box[:, 2] - box[:, 0]
	width = box[:, 3] - box[:, 1]
	center_y = box[:, 0] + 0.5 * height
	center_x = box[:, 1] + 0.5 * width

	gt_height = gt_box[:, 2] - gt_box[:, 0]
	gt_width = gt_box[:, 3] - gt_box[:, 1]
	gt_center_y = gt_box[:, 0] + 0.5 * gt_height
	gt_center_x = gt_box[:, 1] + 0.5 * gt_width

	dy = (gt_center_y - center_y) / height
	dx = (gt_center_x - center_x) / width
	dh = torch.log(gt_height / height)
	dw = torch.log(gt_width / width)

	result = torch.stack([dy, dx, dh, dw], dim=1)
	return result

################################################################################
#   Dataset
################################################################################

class Dataset(object):
	"""The base class for dataset classes.
	To use it, create a new class that adds functions specific to the dataset
	you want to use. For example:

	class CatsAndDogsDataset(Dataset):
		def load_cats_and_dogs(self):
			...
		def load_mask(self, image_id):
			...
		def image_reference(self, image_id):
			...
	"""

	def __init__(self, class_map=None):
		self._image_ids = []

	def load_t2_image(self, image_id, dataset, config, mode):
		"""Load the specified image and return a [H,W] Numpy array.
		"""
		# CHANGE LATER: Update slice index
		self._slice_idx = random.randint(22,40)

		# Get indices
		nb_mods = len(config.MODALITIES)
		t2_idx = int((image_id) * nb_mods + 1) # + 2
		
		t2_file = join(dataset._dir, dataset._image_list[t2_idx])
		t2, opts = nrrd.read(t2_file)

		if mode == 'train' or mode == 'val':
			t2 = np.asarray(t2)[:,:,self._slice_idx]
		
		return t2

	def load_uncertainty(self, image_id, dataset, config, mode):
		"""Load the specified image's associated uncertainty measures and return 4 [H,W] Numpy arrays.
		"""
		# Get indices
		nb_mods = len(config.MODALITIES)

		# Offset is 4 for entropy, 5 for MC variance, 6 for mutual information, 7 for predictive prog
		uncmcvar_idx = int((image_id) * nb_mods + 4) # + 5
		uncmcvar_file = join(dataset._dir, dataset._image_list[uncmcvar_idx])
		uncmcvar, opts = nrrd.read(uncmcvar_file)

		if mode == 'train' or mode == 'val':
			uncmcvar = np.asarray(uncmcvar)[:,:,self._slice_idx]
		
		return uncmcvar

	def load_masks(self, image_id, dataset, config, mode):
		"""Load lesion masks for the given image (ground truth and network output)
		"""
		# Get indices
		nb_mods = len(config.MODALITIES)
		net_mask_idx = int((image_id) * nb_mods ) # + 1
		gt_mask_idx = int((image_id) * nb_mods + 2) # + 3
		net_mask_file = join(dataset._dir, dataset._image_list[net_mask_idx])
		gt_mask_file = join(dataset._dir, dataset._image_list[gt_mask_idx])
		net_mask, opts = nrrd.read(net_mask_file)
		gt_mask, opts = nrrd.read(gt_mask_file)
		gt_mask = np.asarray(gt_mask)
		net_mask = np.asarray(net_mask)
	
		if mode == 'train' or mode == 'val':

			net_mask = net_mask[:,:,self._slice_idx]
			gt_mask = gt_mask[:,:,self._slice_idx]

			# Remove small lesions
			gt_mask, class_ids = remove_tiny_les(gt_mask, nvox=1)
		
			# Return a mask for each lesion instance
			labels, nles = ndimage.label(gt_mask)
			gt_masks = np.zeros([nles, gt_mask.shape[0], gt_mask.shape[1]], dtype=np.int32)

			# Check if there are no lesions

			if nles == 0:
				gt_masks = np.zeros([1, gt_mask.shape[0], gt_mask.shape[1]], dtype=np.int32)
				gt_masks[0] = gt_mask

			# Look for all the voxels associated with a particular lesion

			for i in range(1, nles + 1):
		
				gt_mask[labels != i] = 0
				gt_mask[labels == i] = 1
				gt_masks[i-1] = gt_mask

			gt_masks = gt_masks.transpose(1, 2, 0)
			return net_mask, gt_masks, class_ids, nles

		else:

			# Remove small lesions
			gt_mask, class_ids = remove_tiny_les(gt_mask, nvox=2)
		
			# Return a mask for each lesion instance
			labels, nles = ndimage.label(gt_mask)
			gt_masks = np.zeros([nles, gt_mask.shape[0], gt_mask.shape[1], gt_mask.shape[2]], dtype=np.int32)

			# Check if there are no lesions

			if nles == 0:
				gt_masks = np.zeros([1, gt_mask.shape[0], gt_mask.shape[1], gt_mask.shape[2]], dtype=np.int32)
				gt_masks[0] = gt_mask

			# Look for all the voxels associated with a particular lesion

			for i in range(1, nles + 1):
		
				gt_mask[labels != i] = 0
				gt_mask[labels == i] = 1
				gt_masks[i-1] = gt_mask

			gt_masks = gt_masks.transpose(1, 2, 3, 0)
			return net_mask, gt_masks, class_ids, nles


##############################################################
# Formatting
##############################################################

def resize_image(image, min_dim=None, max_dim=None, padding=False, dims=2):
	"""
	Resizes an image keeping the aspect ratio.

	min_dim: if provided, resizes the image such that it's smaller
		dimension == min_dim
	max_dim: if provided, ensures that the image longest side doesn't
		exceed this value.
	padding: If true, pads image with zeros so it's size is max_dim x max_dim
	dims: The dimension of brain images being used (default is 2 for now i.e. slices)
	Returns:
	
	image: the resized image
	window: (y1, x1, y2, x2). If max_dim is provided, padding might
		be inserted in the returned image. If so, this window is the
		coordinates of the image part of the full image (excluding
		the padding). The x2, y2 pixels are not included.
	scale: The scale factor used to resize the image
	padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
	"""
	# Default window (y1, x1, y2, x2) and default scale == 1.
	h, w = image.shape[:2]
	window = (0, 0, h, w)
	scale = 1

	# Scale?
	if min_dim:
		# Scale up but not down
		scale = max(1, min_dim / min(h, w))
	# Does it exceed max dim?
	if max_dim:
		image_max = max(h, w)
		if round(image_max * scale) > max_dim:
			scale = max_dim / image_max
	# Resize image and mask
	if scale != 1:
		image = transform.resize(
			image, (round(h * scale), round(w * scale)))

	image = image.transpose(1,2,0)

	# Need padding?
	if padding:
		# Get new height and width
		h, w = image.shape[:2]
		top_pad = (max_dim - h) // 2
		bottom_pad = max_dim - h - top_pad
		left_pad = (max_dim - w) // 2
		right_pad = max_dim - w - left_pad
		if dims == 2:
			padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
		else:
			padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
			# CHANGE later - padding = [(top_pad, bottom_pad), (left_pad, right_pad), (front_pad,  back_pad), (0,0)]
		image = np.pad(image, padding, mode='constant', constant_values=0)
		window = (top_pad, left_pad, h + top_pad, w + left_pad)
	
	return image, window, scale, padding


def resize_mask(mask, scale, padding, dims):
	"""Resizes a mask using the given scale and padding.
	Typically, you get the scale and padding from resize_image() to
	ensure both, the image and the mask, are resized consistently.

	scale: mask scaling factor
	padding: Padding to add to the mask in the form
			[(top, bottom), (left, right), (0, 0)]
	"""

	if dims == 2:
		mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
	else:
		# MODIFY later to ensure proper scaling
		mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, scale, 1], order=0)

	mask = np.pad(mask, padding, mode='constant', constant_values=0)
	return mask


def minimize_mask(bbox, mask, mini_shape, dim):
	"""Resize masks to a smaller version to cut memory load.
	Mini-masks can then resized back to image scale using expand_masks()
	"""
	labels = {}
	nles = {}
	labels, nles = ndimage.label(mask)

	if nles > 0:
		mini_mask = np.zeros(mini_shape + (nles,), dtype=bool)
	else:
		mini_mask = np.zeros(mini_shape + (1,), dtype=bool)

	for i in range(nles):	
		mask[labels != i] = 0
		mask[labels == i] = 1
		m = mask[:,:,i]
		y1, x1, y2, x2 = bbox[i][:4]
		m = m[y1:y2, x1:x2]
		if m.size != 0:
			m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
			mini_mask[:, :, i] = np.where(m >= 128, 1, 0)

	return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
	"""Resizes mini masks back to image size. Reverses the change
	of minimize_mask().
	"""
	mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
	for i in range(mask.shape[-1]):
		m = mini_mask[:, :, i]
		y1, x1, y2, x2 = bbox[i][:4]
		h = y2 - y1
		w = x2 - x1
		m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
		mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
	return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
	pass


def unmold_mask(mask, bbox, image_shape):
	"""Converts a mask generated by the neural network into a format similar
	to it's original shape.
	mask: [height, width] of type float. A small, typically 28x28 mask.
	bbox: [y1, x1, y2, x2]. The box to fit the mask in.

	Returns a binary mask with the same size as the original image.
	"""
	threshold = 0.5
	y1, x1, y2, x2 = bbox
	mask = scipy.misc.imresize(
		mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
	mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

	# Put the mask in the right location.
	full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
	full_mask[y1:y2, x1:x2] = mask
	return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
	"""
	scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
	ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
	shape: [height, width] spatial shape of the feature map over which
			to generate anchors.
	feature_stride: Stride of the feature map relative to the image in pixels.
	anchor_stride: Stride of anchors on the feature map. For example, if the
		value is 2 then generate anchors for every other feature map pixel.
	"""
	# Get all combinations of scales and ratios
	scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
	scales = scales.flatten()
	ratios = ratios.flatten()

	# Enumerate heights and widths from scales and ratios
	heights = scales / np.sqrt(ratios)
	widths = scales * np.sqrt(ratios)

	# Enumerate shifts in feature space
	shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
	shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
	shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

	# Enumerate combinations of shifts, widths, and heights
	box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
	box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

	# Reshape to get a list of (y, x) and a list of (h, w)
	box_centers = np.stack(
		[box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
	box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

	# Convert to corner coordinates (y1, x1, y2, x2)
	boxes = np.concatenate([box_centers - 0.5 * box_sizes,
							box_centers + 0.5 * box_sizes], axis=1)

	return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
							 anchor_stride):
	"""Generate anchors at different levels of a feature pyramid. Each scale
	is associated with a level of the pyramid, but each ratio is used in
	all levels of the pyramid.

	Returns:
	anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
		with the same order of the given scales. So, anchors of scale[0] come
		first, then anchors of scale[1], and so on.
	"""
	# Anchors
	anchors = []
	for i in range(len(scales)):
		anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
										feature_strides[i], anchor_stride))
	return np.concatenate(anchors, axis=0)


