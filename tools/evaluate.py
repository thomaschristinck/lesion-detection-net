import numpy as np
import datetime
import time
from collections import defaultdict
import copy
import utils
from scipy import ndimage
import torch

def count_lesions(netseg, target, thresh):
    """
    Connected component analysis of between prediction `h` and ground truth `t` across lesion bin sizes.
    :param netseg: network output on range [0,1], shape=(NxMxO)
    :type netseg: float16, float32, float64
    :param target: ground truth labels, shape=(NxMxO)
    :type target: int16
    :param thresh: threshold to binarize prediction `h`
    :type thresh: float16, float32, float64
    :return: dict

    **************** Courtesy of Tanya Nair ********************
    """

    netseg[netseg >= thresh] = 1
    netseg[netseg < thresh] = 0
    netseg = netseg[0]
    target = target[0]
    mask_target = np.zeros((target.shape[1], target.shape[2], target.shape[3]))
    for lesion in range(target.shape[0]):
        mask_target += target[lesion]

    # To Test netseg = gt_mask (should get ROC as tpr = 1 and fdr = 0 everywhere)
    target, _ = utils.remove_tiny_les(mask_target, nvox=2)
    netseg0 = netseg.copy()
    netseg = ndimage.binary_dilation(netseg, structure=ndimage.generate_binary_structure(3, 2))
    labels = {}
    nles = {}
    labels['target'], nles['target'] = ndimage.label(target)
    labels['netseg'], nles['netseg'] = ndimage.label(netseg)
    found_h = np.ones(nles['netseg'], np.int16)
    ntp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfn = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nb_les = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nles_gt = {'all': nles['target'], 'small': 0, 'med': 0, 'large': 0}

    # Go through ground truth segmentation masks and count true positives/false negatives
    for i in range(1, nles['target'] + 1):
        gt_lesion_size = np.sum(target[labels['target'] == i])
        nles_gt[utils.get_lesion_bin(gt_lesion_size)] += 1
        # List of detected lesions in this area
        h_lesions = np.unique(labels['netseg'][labels['target'] == i])
        # All the voxels in this area contribute to detecting the lesion
        nb_overlap = netseg[labels['target'] == i].sum()
        nb_les[utils.get_lesion_bin(gt_lesion_size)] += 1
        if nb_overlap >= 3 or nb_overlap >= 0.5 * gt_lesion_size:
            ntp[utils.get_lesion_bin(gt_lesion_size)] += 1
            for h_lesion in h_lesions:
                if h_lesion != 0:
                    found_h[h_lesion - 1] = 0
        else:
            nfn[utils.get_lesion_bin(gt_lesion_size)] += 1

    for i in range(1, nles['netseg'] + 1):
        nb_vox = np.sum(netseg0[labels['netseg'] == i])
        if found_h[i - 1] == 1:
            nfp[utils.get_lesion_bin(nb_vox)] += 1

    nb_les['all'] = nb_les['small'] + nb_les['med'] + nb_les['large']
    ntp['all'] = ntp['small'] + ntp['med'] + ntp['large']
    nfp['all'] = nfp['small'] + nfp['med'] + nfp['large']
    nfn['all'] = nfn['small'] + nfn['med'] + nfn['large']

    #print('Number of lesions : ', nb_les)
    #print('Number true positives : ', ntp)
    #print('Number of false positives : ', nfp)
    #print('Number of false negatives : ', nfn)

    tpr = {}
    fdr = {}
    for s in ntp.keys():
        # tpr (sensitivity)
        if nb_les[s] != 0:
            tpr[s] = ntp[s] / nb_les[s]
        elif nb_les[s] == 0 and ntp[s] == 0:
            tpr[s] = 1
        else:
            tpr[s] = 0
        # ppv (1-fdr)
        if ntp[s] + nfp[s] != 0:
            ppv = ntp[s] / (ntp[s] + nfp[s])
        elif ntp[s] == 0:
            ppv = 1
        else:
            ppv = 0
        fdr[s] = 1 - ppv
 
    return {'ntp': ntp, 'nfp': nfp, 'nfn': nfn, 'fdr': fdr, 'tpr': tpr, 'nles': nb_les, 'nles_gt': nles_gt}

def count_boxed_lesions(netbox, target, thresh, scores):
    """
    Connected component analysis of between bounding box prediction `netbox' and ground truth `target` across lesion bin sizes.
    This is a "per slice" analysis
    :param netbox: network output on range [0,1], shape=(lesion_instances x N x M)
    :type netbox: float16, float32, float64
    :param target: ground truth labels, shape=(lwsion_instances x N x M x O)
    :type target: int16
    :param thresh: threshold to binarize prediction `h`
    :type thresh: float16, float32, float64
    :return: dict
    """
    threshed_box_idxs = [x for x, val in enumerate(scores) if val > thresh]
    netbox = netbox[threshed_box_idxs]

    mask_target = np.zeros((target.shape[1], target.shape[2]))
    for lesion in range(target.shape[0]):
        mask_target += target[lesion]
    
    target, _ = utils.remove_tiny_les(mask_target, nvox=2)
    gt_boxes = utils.extract_bboxes(target, dims=2, buf=2)
    
    # To Test netbox = gt_box (should get ROC as tpr = 1 and fdr = 0 everywhere)
    netbox0 = netbox.copy()
    netbox0 = netbox0.astype(int)
    nles = {}
    labels = {}
    labels['target'], nles['target'] = ndimage.label(mask_target)
    nles['netbox'] = netbox.shape[0]
    found_h = np.ones(nles['netbox'], np.int16)
    ntp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfn = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nb_les = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nles_gt = {'all': nles['target'], 'small': 0, 'med': 0, 'large': 0}

    # Go through ground truth boxes and count true positives/false negatives
    for i in range(1, nles['target'] + 1):
        # Find the intersect between gt_boxes for each netbox 
        gt_lesion_size = np.sum(target[labels['target'] == i])
        nles_gt[utils.get_lesion_bin(gt_lesion_size)] += 1
        # List of detected lesions in this area
        h_lesions = []
        box_matrix = np.zeros((mask_target.shape[0], mask_target.shape[1]))
        for j in range(nles['netbox']):
            # Go through each box, get coordinates, and append netbox index to list if it intersects the
            # ground truth lesion 
            y1, x1, y2, x2 = netbox0[j]
            box_matrix[y1:y2, x1:x2] = 1
            intersect = box_matrix[labels['target'] == i].sum()
            print('Intersect & Thresh : ', intersect, thresh)
            if intersect > 0:
                h_lesions.append(j)
            box_matrix[y1:y2, x1:x2] = 0
        # All the voxels in this area contribute to detecting the lesion
        netbox_matrix = np.zeros((mask_target.shape[0], mask_target.shape[1]))
        for i in range(nles['netbox']):
            y1, x1, y2, x2 = netbox0[i]
            netbox_matrix[y1:y2, x1:x2] = 1
        
        nb_overlap = netbox_matrix[labels['target'] == i].sum()

        nb_les[utils.get_lesion_bin(gt_lesion_size)] += 1
        if nb_overlap >= 3 or nb_overlap >= 0.5 * gt_lesion_size:
            ntp[utils.get_lesion_bin(gt_lesion_size)] += 1
            for h_lesion in h_lesions:
                if h_lesion != 0:
                    found_h[h_lesion - 1] = 0
        else:
            nfn[utils.get_box_lesion_bin(gt_lesion_size)] += 1

    for i in range(0, nles['netbox']):
        y1, x1, y2, x2 = netbox0[i]
        netbox_matrix = np.zeros((mask_target.shape[0], mask_target.shape[1]))
        netbox_matrix[y1:y2, x1:x2] = 1
        netbox_size = np.sum(netbox_matrix)
        if found_h[i - 1] == 1:
            nfp[utils.get_box_lesion_bin(netbox_size)] += 1

    nb_les['all'] = nb_les['small'] + nb_les['med'] + nb_les['large']
    ntp['all'] = ntp['small'] + ntp['med'] + ntp['large']
    nfp['all'] = nfp['small'] + nfp['med'] + nfp['large']
    nfn['all'] = nfn['small'] + nfn['med'] + nfn['large']

    tpr = {}
    fdr = {}
    for s in ntp.keys():
        # tpr (sensitivity)
        if nb_les[s] != 0:
            tpr[s] = ntp[s] / nb_les[s]
        elif nb_les[s] == 0 and ntp[s] == 0:
            tpr[s] = 1
        else:
            tpr[s] = 0
        # ppv (1-fdr)
        if ntp[s] + nfp[s] != 0:
            ppv = ntp[s] / (ntp[s] + nfp[s])
        elif ntp[s] == 0:
            ppv = 1
        else:
            ppv = 0
        fdr[s] = 1 - ppv
 
    return {'ntp': ntp, 'nfp': nfp, 'nfn': nfn, 'fdr': fdr, 'tpr': tpr, 'nles': nb_les, 'nles_gt': nles_gt}


def eval_bbox_no_unc(bbox, target, thresh):
    """
    Bounding box evaluation metric between network `h` and ground truth `t` across lesion bin sizes.
    :param h: network output on range [0,1], shape=(NxMxO)
    :type h: float16, float32, float64
    :param t: ground truth labels, shape=(NxMxO)
    :type t: int16
    :param th: threshold to binarize prediction `h`
    :type th: float16, float32, float64
    :return: dict
    """
    # Get data to cpu and find ground truth bounding boxes
    target = target.data.cpu().numpy()
    target, _ = utils.remove_tiny_les(mask_target, nvox=2)
    gt_bbox = extract_bboxes(target, dims=2, buf=2)
    bbox = bbox.data.cpu().numpy()
    print('bbox shape is : ', bbox.shape)
    print('gt bbox shape is : ', gt_bbox.shape)
    nles = {}
    # Go through segmentation masks
    nles['det'] = bbox[0]
    nles['target'] = gt_bbox[0]
    found_h = np.ones(nles['det'], np.int16)
    ntp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfn = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    iou = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nb_les = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nles_gt = {'all': nles['target'], 'small': 0, 'med': 0, 'large': 0}
    # Iterate through each target bounding box
    for i in range(1, nles['target'] + 1):
        gt_lesion_size = utils.get_area(gt_bbox[i])
        nles_gt[utils.get_box_lesion_bin(gt_lesion_size)] += 1
        # List of detected lesions in this area
        h_lesions = []
        max_box_iou = 0
        for lesion in range(1, nles['det'] + 1):
            iou = utils.compute_simple_iou(gt_bbox[i], bbox[lesion])
            if iou > max_box_iou:
                max_box_iou = iou
                h_lesions = lesion 
        # The box with greatest iou (> 0.5) detects the lesion 
        nb_overlap = netseg[labels['target'] == i].sum()
        nb_les[utils.get_box_lesion_bin(gt_lesion_size)] += 1
        # If iou is beyond a threshold we have a tp, otherwise we missed the lesion and have a fn
        if max_box_iou >= 0.4:
            ntp[utils.get_box_lesion_bin(gt_lesion_size)] += 1
            for h_lesion in h_lesions:
                if h_lesion != 0:
                    found_h[h_lesion - 1] = 0
        else:
            nfn[utils.get_lesion_bin(gt_lesion_size)] += 1

    for i in range(1, nles['netseg'] + 1):
        if found_h[i - 1] == 1:
            nb_vox = np.sum(netseg[labels['netseg'] == i])
            nfp[utils.get_box_lesion_bin(nb_vox)] += 1

    nb_les['all'] = nb_les['small'] + nb_les['med'] + nb_les['large']
    ntp['all'] = ntp['small'] + ntp['med'] + ntp['large']
    nfp['all'] = nfp['small'] + nfp['med'] + nfp['large']
    nfn['all'] = nfn['small'] + nfn['med'] + nfn['large']
    print('Number of lesions : ', nb_les)
    print('Number true positives : ', ntp)
    print('Number of false positives : ', nfp)
    print('Number of false negatives : ', nfn)
    tpr = {}
    fdr = {}
    for s in ntp.keys():
        # tpr (sensitivity)
        if nb_les[s] != 0:
            tpr[s] = ntp[s] / nb_les[s]
        elif nb_les[s] == 0 and ntp[s] == 0:
            tpr[s] = 1
        else:
            tpr[s] = 0
        # ppv (1-fdr)
        if ntp[s] + nfp[s] != 0:
            ppv = ntp[s] / (ntp[s] + nfp[s])
        else:
            ppv = 1
        fdr[s] = 1 - ppv
    return {'ntp': ntp, 'nfp': nfp, 'nfn': nfn, 'fdr': fdr, 'tpr': tpr, 'nles': nb_les, 'nles_gt': nles_gt}