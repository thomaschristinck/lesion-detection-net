import numpy as np
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
import visualize

'''
Tanya's Lesion counter - going to use this to create ground truth bounding boxes with classes S, M, L.
'''

def remove_tiny_les(lesion_image, nvox=2):
    labels, nles = ndimage.label(lesion_image)
    for i in range(1, nles + 1):
        nb_vox = np.sum(lesion_image[labels == i])
        if nb_vox <= nvox:
            lesion_image[labels == i] = 0
    return lesion_image


def get_lesion_bin(nvox):
    # Lesion bin - 0 for small lesions, 1 for medium, 2 for large
    if 3 <= nvox <= 10:
        return 0
    elif 11 <= nvox <= 50:
        return 1
    elif nvox >= 51:
        return 2
    else:
        return 0


def bounding_boxes(t, t2, sm_buf, med_buf, lar_buf):
    """
    Bounding box generator for ground truth segmented lesions 't'. Heavily modified from Tanya's 
    lesion_counter.
  
    :param t: ground truth labels, shape=(NxMxO)
    :type t: int16
    :params sm_buf, med_buf, lar_buf: buffers to be applied for respective lesion sizes (voxel padding)
    :return: dict
    """

    # Remove small lesions and init bounding boxes and lesion classes
    t = remove_tiny_les(t, nvox=2)

    labels = {}
    nles = {}
    labels, nles = ndimage.label(t)
    print('Number of lesions', nles)
    boxes = np.zeros([nles + 1, 6], dtype=np.int32)
    classes = np.zeros([nles + 1, 1], dtype=np.int32)
    nb_lesions = nles

    # Look for all the voxels associated with a particular lesion, then bound on x, y, z axis
    for i in range(1, nles + 1):
        
       
        t[labels != i] = 0
        t[labels == i] = 1
 
        # Now we classify the lesion and apply a buffer based on the lesion class (CHANGE LATER??)
        lesion_size = np.sum(t[labels == i])
        classes[i, 0] = get_lesion_bin(lesion_size)
        
        x_indicies = np.where(np.any(t, axis=0))[0]
        y_indicies = np.where(np.any(t, axis=1))[0]
        z_indicies =[]
        for lesion_slice in range(t.shape[-1]):
            if np.any(t[...,lesion_slice]):
                z_indicies.append(lesion_slice)
        z_indicies = np.asarray(z_indicies)
   
        if x_indicies.shape[0]:
            x1, x2 = x_indicies[[0, -1]]
            y1, y2 = y_indicies[[0, -1]]
            z1, z2 = z_indicies[[0, -1]]
            x2 += 1
            y2 += 1
            z2 += 1
            if classes[i] == 0:
                x1 -= sm_buf; x2 += sm_buf; y1 -= sm_buf; y2 += sm_buf; z1 -= sm_buf; z2 += sm_buf
            elif classes[i] == 1:
                x1 -= med_buf; x2 += med_buf; y1 -= med_buf; y2 += med_buf; z1 -= med_buf; z2 += med_buf
            else:
                x1 -= lar_buf; x2 += lar_buf; y1 -= lar_buf; y2 += lar_buf; z1 -= lar_buf; z2 += lar_buf
        else:
            # No mask for this instance
            x1, x2, y1, y2, z1, z2 = 0, 0, 0, 0, 0, 0
       
        boxes[i] = np.array([y1, x1, y2, x2, z1, z2])

    # Reset ground truth mask and then we can draw boxes
    for i in range(1, nb_lesions + 1):
        t[labels == i] = 1
    
    
    for brain_slice in range(18, 50):
        for les in range(1, nles + 1):
            if brain_slice in range(boxes[les, 4], boxes[les, 5]):

                #img2 = visualize.draw_boxes(t2[:,:,35], boxes=boxes[les, 0:4], masks=t[:,:,35])

                if classes[les] == 0:
                    img = visualize.draw_box(t[:,:,brain_slice], boxes[les, 0:4], color=0.2)
                elif classes[les] == 1:
                    img = visualize.draw_box(t[:,:,brain_slice], boxes[les, 0:4], color=0.4)
                else:
                    img = visualize.draw_box(t[:,:,brain_slice], boxes[les, 0:4], color=0.7)
            else:
                img = t[:,:,brain_slice]
  
        imgplt = plt.imshow(img)
        print(brain_slice)
        plt.show()
        plt.savefig('slice.pdf')
    return {'classes': classes, 'boxes': boxes}


