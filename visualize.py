"""
Mask R-CNN
Display and Visualization Functions.

Plot loss function and random colors generator are from Matterport Mask R-CNN:
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import itertools
import colorsys
import numpy as np
import imageio
from skimage.measure import find_contours
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from os.path import join
import utils


############################################################
#  Visualization
############################################################

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.8
    hsv = [(0.88, 0.67, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    image = np.where(mask == 1, image * (1 - alpha) + alpha * color[0] * 255, image)
    return image

def display_instances(image, target, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image[:,:,0].copy() #astype(np.uint32)

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{:.3f}".format(score)
        ax.text(x1, y1 + 8, caption,
                color='tab:gray', size=6, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)
        masked_image = apply_mask(masked_image, target, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image, cmap=plt.cm.pink)

def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 1, x1:x2] = color
    image[y2:y2 + 1, x1:(x2+1)] = color
    image[y1:y2, x1:x1 + 1] = color
    image[y1:y2, x2:x2 + 1] = color
    return image

def get_mask_color(mask, gt_masks, gt_labels, gt_nles):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    max_intersect = 0
    gt_idx_list = []
    for i in range(1, gt_nles + 1):
        # Go through each lesion and see if max intersect is > 0.5 * lesion size, or > 3 voxels
        # if it is, the instance mask is green (TP), otherwise, it is blue (undetected FN).
        lesion_size = np.sum(gt_masks[gt_labels == i])
        intersect = mask[gt_labels == i].sum()
        if intersect >= 3 or intersect >= 0.5 * lesion_size:
            max_intersect = intersect
            gt_idx_list.extend([i])

    if max_intersect != 0:
        # Color mask green; it is a true positive
        hsv = [(0.38, 0.88, 0.8)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        color = colors[0]
        return color, gt_idx_list

    else:
        # Color mask red; it is a false positive
        hsv = [(0.02, 0.67, 1.0)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        color = colors[0]
        gt_idx_list.extend([None])
        return color, gt_idx_list

def get_box_color(box, gt_masks, gt_labels, gt_nles):
    # Look at box. Return label ids of gt lesions with intersect > 3 or 0.5 * lesion size
    # then gt lesions whose labels haven't been returned are false negatives
    y1, x1, y2, x2 = box
    box_matrix = np.zeros((gt_masks.shape[0], gt_masks.shape[1]))
    box_matrix[y1:y2,x1:x2] = 1
    box_size = np.sum(box_matrix)

    # Now for the union, figure out which lesion is contributing to intersect, then 
    # union is the sum of the lesion size plus box size - minus the intersect
    max_intersect = 0
    max_size = 0
    gt_idx_list = []
    for i in range(1, gt_nles + 1):
        
        gt_masks[gt_labels != i] = 0
        gt_masks[gt_labels == i] = 1
 
        # Now we classify the lesion and apply a buffer based on the lesion class (CHANGE LATER??)
        lesion_size = np.sum(gt_masks[gt_labels == i])
        lesion_intersect = np.sum(gt_masks * box_matrix)
        if (lesion_intersect >= 0.5 * lesion_size or lesion_intersect >= 3):
            gt_idx_list.extend([i])
            if lesion_intersect > max_intersect:
                max_intersect = lesion_intersect
                max_size = lesion_size
       
    
    # Then IoU:
    union = box_size + max_size - max_intersect
    iou = max_intersect / union

    # Reset mask
    for i in range(1, gt_nles + 1):
        gt_masks[gt_labels == i] = 1

    if max_intersect != 0:
        # True positive
        color = 'xkcd:bright green'
        return color, gt_idx_list
    else:
        # False negative
        color = 'r'
        gt_idx_list.extend([None])
        return color, gt_idx_list

def draw_unc_mask(t2, unc):
    _, ax = plt.subplots(1, figsize=(12, 12))

    # Show area outside image boundaries.
    ax.axis('off')
    ax.imshow(t2, cmap=plt.cm.gray, interpolation='none')
    ax.imshow(unc, cmap=plt.cm.afmhot, alpha=0.7, interpolation='none')
    return ax


def draw_boxes(image, boxes=None, refined_boxes=None,
               mask=None, gt_mask=None, captions=None, visibilities=None,
               title="", ax=None, pn_labels=False):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    N = boxes.shape[0] if boxes is not None else 0

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Show area outside image boundaries.
    ax.axis('off')

    # Setup image - if no boxes then return the given image as matplotlib axis object
    ax.set_title(title)
    masked_image = image * 255
    masked_image = masked_image.astype(np.uint32).copy()

    #if boxes is None:
        #ax.imshow(masked_image.astype(np.uint32), cmap=plt.cm.gray_r)
        #return ax

    # Generate random colors
    colors = random_colors(N, bright=True)
    gt_idx_list = []
    gt_labels, gt_nles = ndimage.label(gt_mask)

    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 2
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]

            # Get bounding box edge color based on IoU
            if pn_labels:
                bx_color, gt_idx = get_box_color([y1,x1,y2,x2], gt_mask, gt_labels, gt_nles)
                gt_idx_list.extend(gt_idx)

                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=bx_color, facecolor='none')
            else:
                 p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor='r', facecolor='none')

            ax.add_patch(p)

        # Captions
        if captions is not None:
            caption = captions[i]
            caption = np.around(caption, decimals=3)
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            x = random.randint(x1, (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=8, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.4,
                          'pad': 1, 'edgecolor': 'none'})

    # Now go through the gt_idx list and get indices for lesions not detected (FNs):
    fn_indices = [gt_idx for gt_idx in range(1, gt_nles + 1) if gt_idx not in gt_idx_list]
    if len(fn_indices) != 0 and boxes is not None:
        for i in fn_indices:
            gt_mask[gt_labels != i] = 0
            gt_mask[gt_labels == i] = 1
            bbox = utils.extract_bboxes(gt_mask, dims=2, buf=1)
            bbox = bbox[0, 0:4]
            y1, x1, y2, x2 = bbox
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=alpha, linestyle=style,
                                edgecolor='b', facecolor='none')
            ax.add_patch(p)
        for i in range(1, gt_nles + 1):
            gt_mask[gt_labels == i] = 1
    else:
        pass

    #------------------------Masks------------------------------
    # Two options: can display masks as one mask with one color, or can display them as
    # TPs = green, FPs = red, FNs = blue

    if mask is not None and pn_labels == False:
        colors = random_colors(2, bright=True)
        color = colors[0]
        masked_image = apply_mask(masked_image, mask, color)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor=color, edgecolor=color)
            ax.add_patch(p)

    elif mask is not None and pn_labels == True:
            # Return a mask for each lesion instance
            gt_labels, gt_nles = ndimage.label(gt_mask)
            labels, nles = ndimage.label(mask)
            masks = np.zeros([nles, gt_mask.shape[0], gt_mask.shape[1]], dtype=np.int32)

            # Check if there are no lesions

            if nles == 0:
                masks = np.zeros([1, gt_mask.shape[0], gt_mask.shape[1]], dtype=np.int32)
                masks[0] = mask

            # Look for all the voxels associated with a particular lesion

            for i in range(1, nles + 1):
                mask[labels != i] = 0
                mask[labels == i] = 1
                masks[i-1] = mask
            for i in range(1, nles + 1):
                mask[labels == i] = 1

            gt_idx_list = []
            for i in range(masks.shape[0]):
                mask_color, gt_idx = get_mask_color(masks[i], gt_mask, gt_labels, gt_nles)
                gt_idx_list.extend(gt_idx)
                masked_image = apply_mask(masked_image, masks[i], mask_color)
                # Mask Polygon
                # Pad to ensure proper polygons for masks that touch image edges.
                padded_mask = np.zeros(
                    (masks[i].shape[0] + 2, masks[i].shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = masks[i]
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = Polygon(verts, facecolor=mask_color, edgecolor=mask_color)
                    ax.add_patch(p)

            # Now go through the gt_idx list and get indices for lesions not detected (FNs):
            fn_indices = [gt_idx for gt_idx in range(1, gt_nles + 1) if gt_idx not in gt_idx_list]
            if len(fn_indices) != 0:
                for i in fn_indices:
                    gt_mask[gt_labels != i] = 0
                    gt_mask[gt_labels == i] = 1
                    masked_image = apply_mask(masked_image, gt_mask, mask_color)
                    # Mask Polygon
                    # Pad to ensure proper polygons for masks that touch image edges.
                    padded_mask = np.zeros(
                        (gt_mask.shape[0] + 2, gt_mask.shape[1] + 2), dtype=np.uint8)
                    padded_mask[1:-1, 1:-1] = gt_mask
                    contours = find_contours(padded_mask, 0.5)
                    for verts in contours:
                        # Subtract the padding and flip (y, x) to (x, y)
                        verts = np.fliplr(verts) - 1
                        p = Polygon(verts, facecolor='b', edgecolor='b')
                        ax.add_patch(p)
                for i in range(1, gt_nles + 1):
                    gt_mask[gt_labels == i] = 1
            else:
                pass

    ax.imshow(masked_image.astype(np.uint32), cmap=plt.cm.gray)
    return ax

def plot_loss(loss, val_loss, save=True, log_dir=None):
    loss = np.array(loss)
    val_loss = np.array(val_loss)

    plt.figure("loss")
    plt.gcf().clear()
    plt.plot(loss[:, 0], label='train')
    plt.plot(val_loss[:, 0], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("rpn_class_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 1], label='train')
    plt.plot(val_loss[:, 1], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "rpn_class_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("rpn_bbox_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 2], label='train')
    plt.plot(val_loss[:, 2], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "rpn_bbox_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("mrcnn_class_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 3], label='train')
    plt.plot(val_loss[:, 3], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "mrcnn_class_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("mrcnn_bbox_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 4], label='train')
    plt.plot(val_loss[:, 4], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "mrcnn_bbox_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("mrcnn_mask_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 5], label='train')
    plt.plot(val_loss[:, 5], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "mrcnn_mask_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

class IndexTracker(object):
    def __init__(self, ax, fig, X, Y, Z, W, U):
        self.ax = ax
        self.fig = fig
        plt.subplots_adjust(top=0.9)
        self.ax[0][0].axis('off')
        self.ax[0][1].axis('off')
        self.ax[0][2].axis('off')
        self.ax[1][0].axis('off')
        self.ax[1][1].axis('off')
        self.ax[1][2].axis('off')

        fontdict = {'fontsize':10}
        #self.ax[0][0].set_title(r'Tanyas netseg', fontdict=fontdict)
        #self.ax[0][1].set_title(r'My netseg', fontdict=fontdict)
        #self.ax[1][0].set_title(r'U-Net Output ($\sigma=0.5$)', fontdict=fontdict)
        #self.ax[1][1].set_title(r'U-Net MC Sample Variance', fontdict=fontdict)
        self.U = U
        self.W = W
        self.X = X
        self.Y = Y
        self.Z = Z
        x_rows, x_cols,  self.slices, x_colors = X.shape
        self.ind = self.slices//2
        self.im1 = ax[0][0].imshow(self.X[:,:, self.ind,:])
        self.im2 = ax[0][1].imshow(self.Y[:,:, self.ind, :])
        self.im3 = ax[0][2].imshow(self.Z[:,:, self.ind, :])
        self.im4 = ax[1][0].imshow(self.W[:,:, self.ind, :])
        self.im5 = ax[1][1].imshow(self.U[:,:, self.ind], cmap=plt.cm.gray)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices


        self.update()

    def update(self):
        plt.subplots_adjust(top=0.9)
        self.im1.set_data(self.X[:,:,self.ind])
        self.im2.set_data(self.Y[:,:,self.ind])
        self.im3.set_data(self.Z[:,:,self.ind])
        self.im4.set_data(self.W[:,:,self.ind])
        self.im5.set_data(self.U[:,:,self.ind])
        self.ax[0][0].set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw_idle()
        self.im2.axes.figure.canvas.draw_idle()
        self.im3.axes.figure.canvas.draw_idle()
        self.im4.axes.figure.canvas.draw_idle()
        self.im5.axes.figure.canvas.draw_idle()

def scroll_display(image1, image2, image3, image4, image5, figsize):
    fig, ax = plt.subplots(2,3, figsize=(20,20))
    tracker = IndexTracker(ax, fig, image1, image2, image3, image4, image5)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.tight_layout()
    plt.show()

def build_image3d(t2, target, netseg, unc, threshed, model, class_names, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Iterate through image slices of depth t2.shape[2]
    depth = t2.shape[2]
    fig, ax = plt.subplots(2,2, figsize=(20,20))
   
    ax[0][0].axis('off')
    ax[0][1].axis('off')
    ax[1][0].axis('off')
    ax[1][1].axis('off')

    for idx in range(depth):
        target_slice = target[:,:,idx]
        threshed_slice = threshed[:,:,idx]
        netseg_slice = netseg[:,:,idx]
        t2_slice = t2[:,:,idx]
        unc_slice = unc[:,:,idx]
        
        # Stack slices to make the input image
        image_slice = np.stack([t2_slice, unc_slice, netseg_slice], axis = 0)

        results = model.detect([image_slice])
        r = results[0]
        boxes = r['rois']
        masks = r['masks']
        class_ids = r['class_ids'] 
        scores =  r['scores']
        #scores = np.around(scores, decimals=2)
        image_slice = image_slice.transpose(1,2,0)

        # Number of instances
        N = boxes.shape[0]
        _, nles_nseg = ndimage.label(threshed_slice)
        _, nles_gt = ndimage.label(threshed_slice)

        # Generate bounding box slices
        if N != 0:
            assert N == masks.shape[-1] == class_ids.shape[0]
            #ax[0][0] = draw_boxes(t2_slice, mask=netseg_slice, gt_mask=target_slice, pn_labels=True)
            ax[0][0] = draw_boxes(t2_slice, boxes=boxes, gt_mask=target_slice, pn_labels=True)
        else:
            ax[0][0] = draw_boxes(t2_slice) 

        plt.savefig(join('/usr/local/data/thomasc/outputs/det_net', str(idx) + '.png'), bbox_inches='tight')
        plt.close('all')

        if nles_nseg > 0 or nles_gt > 0:
            ax[0][0] = draw_boxes(t2_slice, gt_mask=target_slice, mask=threshed_slice, pn_labels=True)
        else:
            ax[0][0] = draw_boxes(t2_slice)     

        plt.savefig(join('/usr/local/data/thomasc/outputs/det_net', str(idx) + str(idx) + '.png'), bbox_inches='tight')
        plt.close('all')

        ax[0][0] = draw_boxes(t2_slice, mask=target_slice)  

        plt.savefig(join('/usr/local/data/thomasc/outputs/det_net', 'target' + str(idx) + '.png'), bbox_inches='tight')
        plt.close('all')

        ax[0][0] = draw_unc_mask(t2_slice, unc_slice)  

        plt.savefig(join('/usr/local/data/thomasc/outputs/det_net', 'unc' + str(idx) + '.png'), bbox_inches='tight')
        plt.close('all')

    print('---------- Reading images into numpy array ------------')

    im = imageio.imread(join('/usr/local/data/thomasc/outputs/det_net', str(idx) + '.png'), as_gray=False, pilmode="RGB")
    boxed_image = np.zeros([depth, im.shape[0], im.shape[1], im.shape[2]])
    nseg_image = np.zeros([depth, im.shape[0], im.shape[1], im.shape[2]])
    target_image = np.zeros([depth, im.shape[0], im.shape[1], im.shape[2]])
    unc_image = np.zeros([depth, im.shape[0], im.shape[1], im.shape[2]])

    for idx in range(depth):
        boxed_image[idx] = imageio.imread(join('/usr/local/data/thomasc/outputs/det_net', str(idx) + '.png'), as_gray=False, pilmode="RGB")
        nseg_image[idx] = imageio.imread(join('/usr/local/data/thomasc/outputs/det_net', str(idx) + str(idx) + '.png'), as_gray=False, pilmode="RGB")
        target_image[idx] = imageio.imread(join('/usr/local/data/thomasc/outputs/det_net', 'target' + str(idx) + '.png'), as_gray=False, pilmode="RGB")
        unc_image[idx] = imageio.imread(join('/usr/local/data/thomasc/outputs/det_net', 'unc' + str(idx) + '.png'), as_gray=False, pilmode="RGB")
    plt.close('all')
    
    # Images with masks 
    boxed_image = boxed_image.astype(np.uint8)
    nseg_image = nseg_image.astype(np.uint8)
    target_image = target_image.astype(np.uint8)
    unc_image = unc_image.astype(np.uint8)
    boxed_image = boxed_image.transpose(1, 2, 0, 3)
    nseg_image = nseg_image.transpose(1, 2, 0, 3)
    target_image = target_image.transpose(1, 2, 0, 3)
    unc_image = unc_image.transpose(1, 2, 0, 3)
    nseg_image = np.rot90(nseg_image, axes=(0,1))
    boxed_image = np.rot90(boxed_image, axes=(0,1))
    target_image = np.rot90(target_image, axes=(0,1))
    unc_image = np.rot90(unc_image, axes=(0,1))


    # Base images
    target = target * 255
    target = target.astype(np.uint32).copy()
    threshed = threshed * 255
    threshed = threshed.astype(np.uint32).copy()
    target = np.rot90(target, axes=(0,1))
    netseg = np.rot90(netseg, axes=(0,1))
    unc = np.rot90(unc, axes=(0,1))
    threshed = np.rot90(threshed, axes=(0,1))
    t2 = np.rot90(t2, axes=(0,1))

    scroll_display(boxed_image, nseg_image, target_image, unc_image, t2, figsize)

