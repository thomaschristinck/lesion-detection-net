"""
Mask R-CNN
Display and Visualization Functions.

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

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


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

def apply_target(image, target, color, alpha=0.5):
    
    image = np.where(target > 0.5, 0, image)
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

def build_image(image, target, boxes, masks, bunet_mask, class_ids, class_names,
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
        fig, ax = plt.subplots(1, 3, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    for i in range(3):
        ax[i].axis('off')

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
        ax[0].add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{:.3f}".format(score)
        ax[0].text(x1, y1 + 8, caption,
                color='tab:gray', size=6, backgroundcolor="none")

        masked_image = masked_image
        ax[0].imshow(masked_image, cmap=plt.cm.pink)
        ax[1].imshow(target, cmap=plt.cm.pink)
        ax[2].imshow(bunet_mask, cmap=plt.cm.pink)

    plt.show()

def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
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


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")

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
    gt_idx = 0
    for i in range(1, gt_nles + 1):
        # Go through each lesion and see if max intersect is > 0.5 * lesion size, or > 3 voxels
        # if it is, the instance mask is green (TP), otherwise, it is blue (undetected FN).
        lesion_size = np.sum(gt_masks[gt_labels == i])
        intersect = mask[gt_labels == i].sum()
        if intersect >= 3 or intersect >= 0.5 * lesion_size:
            max_intersect = intersect
            gt_idx = i

    N=2
    print('Max intersect : ', max_intersect)
    if max_intersect != 0:
        # Color mask green; it is a true positive
        hsv = [(0.38, 0.88, 0.8) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        color = colors[0]
        return color, gt_idx

    else:
        # Color mask red; it is a false positive
        hsv = [(0.02, 0.67, 1.0) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        color = colors[0]
        return color, None

def get_box_color(box, masks):
    y1, x1, y2, x2 = box
    box_matrix = np.zeros((masks.shape[0], masks.shape[1]))
    box_matrix[y1:y2, x1:x2,] = 1
    box_size = np.sum(box_matrix)
    intersect = np.sum(box_matrix * masks)

    # Now for the union, figure out which lesion is contributing to intersect, then 
    # union is the sum of the lesion size plus box size - minus the intersect
    labels, nles = ndimage.label(masks)
    max_intersect = 0
    max_size = 0
    for i in range(1, nles + 1):
        
        masks[labels != i] = 0
        masks[labels == i] = 1
 
        # Now we classify the lesion and apply a buffer based on the lesion class (CHANGE LATER??)
        lesion_size = np.sum(masks[labels == i])
        lesion_intersect = np.sum(masks * box_matrix)
        if lesion_intersect > max_intersect:
            lesion_choice = i
            max_intersect = lesion_intersect
            max_size = lesion_size
       
    
    # Then IoU:
    union = box_size + max_size - intersect
    iou = intersect / union

    # Reset mask
    for i in range(1, nles + 1):
        masks[labels == i] = 1

    labels, nles2 = ndimage.label(masks)
    if intersect != 0:
        # True positive
        color = 'xkcd:bright green'
    else:
        # False negative
        color = 'r'
    
    return color

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
                bx_color = get_box_color([y1,x1,y2,x2], gt_mask)

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

    
    #------------------------Masks------------------------------
    # Two options: can display masks as one mask with one color, or can display them as
    # TPs = green, FPs = red, FNs = blue

    if mask is not None and pn_labels == False:
        print('Mask shape: ', mask.shape)
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
            print('Nles 1 : ', nles)
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

            print('Masks shape :', masks.shape[0])
            gt_idx_list = []
            for i in range(masks.shape[0]):
                mask_color, gt_idx = get_mask_color(masks[i], gt_mask, gt_labels, gt_nles)
                gt_idx_list.append(gt_idx)
                print('Masks shape (for i in masks shape[0]):', masks.shape[0])
                print('Mask color : ', mask_color)
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

        #fontdict = {'fontsize':10}
        #self.ax[0][0].set_title(r'Det-Net Output, GT Mask, and T2 Image (confidence thresh = 0.95)', fontdict=fontdict)
        #self.ax[0][1].set_title(r'Ground Truth Lesion Mask', fontdict=fontdict)
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
            ax[0][0] = draw_boxes(t2_slice, boxes=boxes, gt_mask=target_slice, pn_labels=True)
            #ax[0][0] = draw_boxes(t2_slice, boxes=boxes, captions=scores)
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

