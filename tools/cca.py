# connected component analysis of segmentation compared to ground truth
import numpy as np
from scipy import ndimage

_thresh = 0.50
_smooth = 1

_LESION_UNC_VA = {
    'bald': {'ymin': -304332.435036, 'ymax': -165.66824439, 'xmin': -287025.719183, 'xmax': -60.0155142638},
    'ent': {'ymin': -264211.01785, 'ymax': -120.603669105, 'xmin': -246545.542156, 'xmax': -42.3070650557},
    'prdvar': {'ymin': -77261.4, 'ymax': -63.0317, 'xmin': -73166.6, 'xmax': -32.3993},
    'varmcs': {'ymin': -357833.52626, 'ymax': -254.615228264, 'xmin': -340539.075182, 'xmax': -92.6999732885}}


def remove_tiny_les(lesion_image, nvox=2):
    labels, nles = ndimage.label(lesion_image)
    for i in range(1, nles + 1):
        nb_vox = np.sum(lesion_image[labels == i])
        if nb_vox <= nvox:
            lesion_image[labels == i] = 0
    return lesion_image


def global_dice(h, t):
    h = h.flatten()
    t = t.flatten()
    intersection = np.sum(h * t)
    union = np.sum(h) + np.sum(t)
    dice = (2. * intersection + _smooth) / (union + _smooth)
    return dice


def get_lesion_bin(nvox):
    if 3 <= nvox <= 10:
        return 'small'
    elif 11 <= nvox <= 50:
        return 'med'
    elif nvox >= 51:
        return 'large'
    else:
        return "small"


def cca_img_bin(h, t, u, uth, metric):
    t = remove_tiny_les(t, nvox=2)
    h0 = h.copy()
    h = ndimage.binary_dilation(h, structure=ndimage.generate_binary_structure(3, 2))
    t_unc_labels = get_unc_labels(t, u, metric, 'y')
    h_unc_labels = get_unc_labels(h0, u, metric, 'x')

    labels = {}
    nles = {}
    labels['h'], nles['h'] = ndimage.label(h)
    labels['t'], nles['t'] = ndimage.label(t)
    found_h = np.ones(nles['h'], np.int16)
    ntp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfn = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nb_les = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nles_gt = {'all': nles['t'], 'small': 0, 'med': 0, 'large': 0}
    for i in range(1, nles['t'] + 1):
        t_unc = np.max(t_unc_labels[labels['t'] == i])
        lesion_size = np.sum(t[labels['t'] == i])
        nles_gt[get_lesion_bin(lesion_size)] += 1
        if t_unc < uth:
            # list of detected lesions in this area
            h_lesions = np.unique(labels['h'][labels['t'] == i])
            # all the voxels in this area contribute to detecting the lesion
            nb_overlap = h[labels['t'] == i].sum()
            nb_les[get_lesion_bin(lesion_size)] += 1
            if nb_overlap >= 3 or nb_overlap >= 0.5 * lesion_size:
                ntp[get_lesion_bin(lesion_size)] += 1
                for h_lesion in h_lesions:
                    if h_lesion != 0:
                        found_h[h_lesion - 1] = 0
            else:
                nfn[get_lesion_bin(lesion_size)] += 1

    for i in range(1, nles['h'] + 1):
        if found_h[i - 1] == 1:
            h_unc = np.max(h_unc_labels[labels['h'] == i])
            if h_unc < uth:
                nb_vox = np.sum(h0[labels['h'] == i])
                nfp[get_lesion_bin(nb_vox)] += 1

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
        else:
            ppv = 1
        fdr[s] = 1 - ppv

    return {'ntp': ntp, 'nfp': nfp, 'nfn': nfn, 'fdr': fdr, 'tpr': tpr, 'nles': nb_les, 'nles_gt': nles_gt}


def cca_img(h, t, u, uth, metric):
    t = remove_tiny_les(t, nvox=2)
    h0 = h.copy()
    h = ndimage.binary_dilation(h, structure=ndimage.generate_binary_structure(3, 2))
    t_unc_labels = get_unc_labels(t, u, metric, 'y')
    h_unc_labels = get_unc_labels(h0, u, metric, 'x')

    labels = {}
    nles = {}
    labels['h'], nles['h'] = ndimage.label(h)
    labels['t'], nles['t'] = ndimage.label(t)
    found_h = np.ones(nles['h'], np.int16)
    ntp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfn = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nb_les = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nles_gt = {'all': nles['t'], 'small': 0, 'med': 0, 'large': 0}
    for i in range(1, nles['t'] + 1):
        t_unc = np.max(t_unc_labels[labels['t'] == i])
        lesion_size = np.sum(t[labels['t'] == i])
        nles_gt[get_lesion_bin(lesion_size)] += 1
        if t_unc < uth:
            # list of detected lesions in this area
            h_lesions = np.unique(labels['h'][labels['t'] == i])
            # all the voxels in this area contribute to detecting the lesion
            nb_overlap = h[labels['t'] == i].sum()
            nb_les[get_lesion_bin(lesion_size)] += 1
            if nb_overlap >= 3 or nb_overlap >= 0.5 * lesion_size:
                ntp[get_lesion_bin(lesion_size)] += 1
                for h_lesion in h_lesions:
                    if h_lesion != 0:
                        found_h[h_lesion - 1] = 0
            else:
                nfn[get_lesion_bin(lesion_size)] += 1

    for i in range(1, nles['h'] + 1):
        if found_h[i - 1] == 1:
            h_unc = np.max(h_unc_labels[labels['h'] == i])
            if h_unc < uth:
                nb_vox = np.sum(h0[labels['h'] == i])
                nfp[get_lesion_bin(nb_vox)] += 1

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
        else:
            ppv = 1
        fdr[s] = 1 - ppv

    return {'ntp': ntp, 'nfp': nfp, 'nfn': nfn, 'fdr': fdr, 'tpr': tpr, 'nles': nb_les, 'nles_gt': nles_gt}


def cca_img_no_unc(h, t, th):
    """
    Connected component analysis of between prediction `h` and ground truth `t` across lesion bin sizes.
    :param h: network output on range [0,1], shape=(NxMxO)
    :type h: float16, float32, float64
    :param t: ground truth labels, shape=(NxMxO)
    :type t: int16
    :param th: threshold to binarize prediction `h`
    :type th: float16, float32, float64
    :return: dict
    """
    h[h >= th] = 1
    h[h < th] = 0
    h = h.astype(np.int16)

    t = remove_tiny_les(t, nvox=2)
    h = ndimage.binary_dilation(h, structure=ndimage.generate_binary_structure(3, 2))

    labels = {}
    nles = {}
    labels['h'], nles['h'] = ndimage.label(h)
    labels['t'], nles['t'] = ndimage.label(t)
    found_h = np.ones(nles['h'], np.int16)
    ntp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfn = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nb_les = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nles_gt = {'all': nles['t'], 'small': 0, 'med': 0, 'large': 0}
    for i in range(1, nles['t'] + 1):
        lesion_size = np.sum(t[labels['t'] == i])
        nles_gt[get_lesion_bin(lesion_size)] += 1
        # list of detected lesions in this area
        h_lesions = np.unique(labels['h'][labels['t'] == i])
        # all the voxels in this area contribute to detecting the lesion
        nb_overlap = h[labels['t'] == i].sum()
        nb_les[get_lesion_bin(lesion_size)] += 1
        if nb_overlap >= 3 or nb_overlap >= 0.5 * lesion_size:
            ntp[get_lesion_bin(lesion_size)] += 1
            for h_lesion in h_lesions:
                if h_lesion != 0:
                    found_h[h_lesion - 1] = 0
        else:
            nfn[get_lesion_bin(lesion_size)] += 1

    for i in range(1, nles['h'] + 1):
        if found_h[i - 1] == 1:
            nb_vox = np.sum(h[labels['h'] == i])
            nfp[get_lesion_bin(nb_vox)] += 1

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
        else:
            ppv = 1
        fdr[s] = 1 - ppv

    return {'ntp': ntp, 'nfp': nfp, 'nfn': nfn, 'fdr': fdr, 'tpr': tpr, 'nles': nb_les, 'nles_gt': nles_gt}


def get_unc_labels(x, unc, metric, xy):
    x_big = ndimage.binary_dilation(x, structure=ndimage.generate_binary_structure(3, 2))
    labels, nles = ndimage.label(x_big)
    unc_labels = np.zeros_like(unc)
    for i in range(1, nles + 1):
        unc_labels[labels == i] = np.sum(np.log(unc[labels == i] + 1e-5))

    unc_labels[unc_labels != 0] = (unc_labels[unc_labels != 0] - _LESION_UNC_VA[metric][xy + 'min']) / (
        _LESION_UNC_VA[metric][xy + 'max'] - _LESION_UNC_VA[metric][xy + 'min'])
    return unc_labels

    
    
def paint_cca_img(h, t):
    TP_COLOUR=1
    FP_COLOUR=2
    FN_COLOUR=3

    h_paint = h.copy() # this should be the un-connected labels

    t = remove_tiny_les(t)

    # get the 18 neighbourhood of the hypothesis
    neighbourhood18 = ndimage.generate_binary_structure(3, 2)
    h = ndimage.binary_dilation(h, structure=neighbourhood18)

    labels = {}
    nles = {}
    labels['h'], nles['h'] = ndimage.label(h)
    labels['t'], nles['t'] = ndimage.label(t)
    ntp = 0; nfp = 0; nfn = 0
    found_h = np.ones(nles['h'], np.int16)

    for i in range(1,nles['t']+1):
        got_les = False
        h_lesions = np.unique(labels['h'][labels['t']==i])
        for h_lesion in h_lesions:
            if h_lesion != 0:
                nb_overlap = h[labels['h']==h_lesion].sum()
                if nb_overlap >= 3 or nb_overlap >= 0.5*np.sum(t[labels['t']==i]):
                    # this is a a true positive
                    h_paint[labels['h']==h_lesion] *= TP_COLOUR
                    found_h[h_lesion-1] = 0
                    got_les = True
        # if we didn't get it --> false negative
        if not got_les:
            h_paint[labels['t']==i] = FN_COLOUR

    # any remaining that we found are false positivse
    for i, fp in enumerate(found_h):
        if fp:
            h_paint[labels['h']==(i+1)] *= FP_COLOUR

    labels_hpaint, nles = ndimage.label(h_paint)
    for i in range(1,nles+1):
        nb_vox = np.size(labels_hpaint[labels_hpaint==i])
        if nb_vox < 3:
            h_paint[labels_hpaint==i] = 0
    return h_paint    


def ohe(x):
    ohe = np.zeros_like(x)
    ohe = np.repeat(np.expand_dims(x,-1),3,-1)
    ohe[:,:,:,0][x!=2] = 0 # red --> FP(2)
    ohe[:,:,:,1][x!=1] = 0 # gre --> TP(1)
    ohe[:,:,:,2][x!=3] = 0 # blu --> FN(3)
    ohe[ohe>0]=1
    return ohe
    
# def cca_img(h, t, thresh, read_img=True):
#     """
#     1. binarize segmentation 'h' based on sigmoid 0.5 thresh
#     2. generate blob (18 neighbourhood) of the 'h'
#     3. remove tiny lesions in truth 't' (<=2 voxels in size)
#     4. count true positives:
#         for lesion in t:
#             overlap  = _get_overlap(h, t)
#             if overap >= voxels:
#                 tp ++
#             else if ( overlap > 0.5 * size(lesion))
#                 tp ++
#             else:
#                 fn ++

#     5. count false positives:
#         for lesion in h:
#             overlap  = _get_overlap(h, t)
#             if overlap == 0:
#                 fp ++
#     """
#     if read_img:
#         h, header = nrrd.read(h)
#         t, header = nrrd.read(t)

#     h[h < thresh] = 0
#     h[h >= thresh] = 1
#     t = t.astype(np.int16)
#     h = h.astype(np.int16)

#     t = remove_tiny_les(t)

#     # get the 18 neighbourhood of the hypothesis
#     neighbourhood18 = ndimage.generate_binary_structure(3, 2)
#     h = ndimage.binary_dilation(h, structure=neighbourhood18)

#     labels = {}
#     nles = {}
#     labels['h'], nles['h'] = ndimage.label(h)
#     labels['t'], nles['t'] = ndimage.label(t)

#     ntp = 0; nfp = 0; nfn = 0
#     found_h = np.ones(nles['h'], np.int16)

#     for i in range(1,nles['t']):
#         got_les = False
#         h_lesions = np.unique(labels['h'][labels['t']==i])
#         for h_lesion in h_lesions:
#             if h_lesion != 0:
#                 nb_overlap = np.size(labels['h']==h_lesion)
#                 if nb_overlap >= 3 or nb_overlap >= 0.5*np.sum(t[labels['t']==i]):
#                     got_les = True
#                     found_h[h_lesion-1] = 0
#         if got_les:
#             ntp += 1
#         else:
#             nfn += 1

#     nfp = np.sum(found_h)

#     # tpr
#     if nles['t']!= 0:
#         tpr = ntp / nles['t']
#     elif nles['t']==0 and ntp ==0:
#         tpr = 1
#     else:
#         tpr = 0

#     # ppv, fdr
#     if ntp+nfp != 0:
#         ppv = ntp / (ntp+nfp)
#     else:
#         ppv = 1

#     # fdr
#     # if ntp+nfp != 0:
#     #     fdr = nfp / (ntp+nfp)
#     # else:
#     #     fdr = 0        

#     # f1 score
#     # denom = (2*ntp + nfp + nfn)
#     # if denom != 0:
#     #     f1_score = 2*ntp / denom
#     # else:
#     #     f1_score = 1

#     return {'ntp':ntp, 'nfp':nfp, 'nfn':nfn, 'ppv':ppv, 'tpr':tpr, 'fdr':1-ppv}
