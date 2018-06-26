from timeit import default_timer as timer
import csv
import utils
import numpy as np
import nrrd
import torch
from torch.autograd import Variable

from tools.unc_metrics import mi_uncertainty, entropy, prd_variance, prd_uncertainty
from os.path import join
import model as modellib

_OPTS = {'space': 'RAS', 'space directions': [(1, 0, 0), (0, 1, 0), (0, 0, 3)]}

'''
Analyzer is essentially Tanya's Bunet Analyzer code modified to work with a pytorch model
'''

class MRCNNAnalyzer:
    def __init__(self, model, config, datagen, out_dir, nb_mc):
        self.__model = model
        self.__config = config
        self.__datagen = datagen
        self.__out_dir = out_dir
        self.__nb_mc = nb_mc

    @staticmethod
    def _get_unc_img(mu_mcs, log_var_mcs):
        bald = bald_uncertainty(sigmoid(mu_mcs))
        ent = entropy(sigmoid(mu_mcs))
        prd_var = prd_variance(log_var_mcs)
        prd_unc = prd_uncertainty(mu_mcs, prd_var)
        return {'bald': bald, 'ent': ent, 'prd_var': prd_var, 'prd_unc': prd_unc}

    @staticmethod
    def _get_prd_stats(y, h):
        stats = cca_img(h, y, 0.5)
        dice = global_dice(h, y)
        stats.update({'dice': dice})
        return stats

    @staticmethod
    def _clip_at_thresh(x, a, thresh):
        x[a >= thresh] = 1
        x[a < thresh] = 0
        return x

    @staticmethod
    def _keep_below_thresh(x, a, thresh):
        x[a >= thresh] = 0
        return x

    def cca(self, out_file, thresh):
        start = timer()
        with open(join(self.__out_dir, out_file), 'w', newline='') as csvfile:
            stats_writer = csv.writer(csvfile, delimiter=',')
            stats_writer.writerow(['subj', 'tp', 'mean_fdr', 'mean_tpr', 'mean_dice', 'mean roi'])
            nb_subj = 0
            total_mean_overlaps = 0
            ustats = {'fdr': 0, 'tpr': 0, 'dice': 0}
    
            for inputs in self._data_gen:
                images = inputs[0]

                # image as numpy array for mc samples
                images = np.repeat(images, self.__nb_mc, 0)

                # Wrap in variables
                images = Variable(images)
        
                # To GPU
                if self.__config.GPU_COUNT:
                    images = images.cuda()

                # Compute the IoUs between the two sets of boxes - in this case we use the maximum IoU for each set
                overlaps = utils.compute_2D_overlaps(results['rois'], gt_boxes)
                max_overlaps_idx = np.zeros(overlaps.shape[1])
                max_overlaps = np.zeros(overlaps.shape[1])
                for i in range(overlaps.shape[1]):
                    max_overlaps_idx[i] = np.argmax(overlaps[:, i], axis=0) 
                    max_overlaps[i] = overlaps[:,i][max_overlaps_idx]
                mean_overlaps = np.mean(max_overlaps)
                total_mean_overlaps += mean_overlaps

                # Run detection
                results = self.__model.detect([images])
                results = np.asarray(results['rois'], np.float32)[..., 0]
                h = sigmoid(np.mean(results, 0))
                print(h)
                y = y[0, ..., 0]
                # Sigmoid thresholding
                h_unc_thresh = self._clip_at_thresh(h, h, thresh)
                stats = self._get_prd_stats(y, h_unc_thresh)
                ustats['fdr'] += stats['fdr']['all']
                ustats['tpr'] += stats['tpr']['all']
                ustats['dice'] += stats['dice']
                nb_subj += 1
                print('completed subject {}     {:.2f}m'.format(nb_subj, (timer() - start) / 60))
                stats_writer.writerow([subj[0], tp[0], stats['fdr']['all'], stats['tpr']['all'], stats['dice'], mean_overlaps])
            stats_writer.writerow(
                ['mean_subj', '_', ustats['fdr'] / nb_subj, ustats['tpr'] / nb_subj, ustats['dice'] / nb_subj, total_mean_overlaps/nb_subj])
        print("completed in {:.2f}m".format((timer() - start) / 60))

    def roc(self, out_file, thresh_start, thresh_stop, thresh_step):
        start = timer()
        with open(join(self.__out_dir, out_file), 'w', newline='') as csvfile:
            stats_writer = csv.writer(csvfile, delimiter=',')
            stats_writer.writerow(['unc_thresh', 'mean_fdr', 'mean_tpr', 'mean_dice'])
            nb_subj = 0
            thresh = np.arange(thresh_start, thresh_stop, thresh_step)
            ustats = {}
            [ustats.update({t: {'fdr': 0, 'tpr': 0, 'dice': 0}}) for t in thresh]
            test_set = modellib.Dataset(self.__dataset, self.__config, augment=True)
            data_gen = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)
            for subj, tp, x, y in data_gen:
                x_mc = np.repeat(x, self.__nb_mc, 0)
                mu_mcs = sess.run(self.__model.predictor,
                                      feed_dict={self.__model.x: x_mc, self.__model.keep_prob: 0.5})
                mu_mcs = np.asarray(mu_mcs, np.float32)[..., 0]
                ent = entropy(sigmoid(mu_mcs))
                h = sigmoid(np.mean(mu_mcs, 0))
                y = y[0, ..., 0]
                for t in thresh:
                    h_unc_thresh = self._keep_below_thresh(h, ent, t)
                    stats = self._get_prd_stats(y, h_unc_thresh)
                    ustats[t]['fdr'] += stats['fdr']['all']
                    ustats[t]['tpr'] += stats['tpr']['all']
                    ustats[t]['dice'] += stats['dice']
                nb_subj += 1
                print('completed subject {}     {:.2f}m'.format(nb_subj, (timer() - start) / 60))
            for t in thresh:
                stats_writer.writerow(
                    [t, ustats[t]['fdr'] / nb_subj, ustats[t]['tpr'] / nb_subj, ustats[t]['dice'] / nb_subj])
        print("completed in {:.2f}m".format((timer() - start) / 60))

    def roc_sigmoid(self, out_file, thresh_start, thresh_stop, thresh_step):
        start = timer()
        with open(join(self.__out_dir, out_file), 'w', newline='') as csvfile:
            stats_writer = csv.writer(csvfile, delimiter=',')
            stats_writer.writerow(['unc_thresh', 'mean_fdr', 'mean_tpr', 'mean_dice'])
            nb_subj = 0
            thresh = np.arange(thresh_start, thresh_stop, thresh_step)
            ustats = {}
            [ustats.update({t: {'fdr': 0, 'tpr': 0, 'dice': 0}}) for t in thresh]
    
            for inputs in data_gen:
                images = inputs[0]
                rpn_match = inputs[1]
                rpn_bbox = inputs[2]
                gt_class_ids = inputs[3]
                gt_boxes = inputs[4]
                gt_masks = inputs[5]
                image_metas = inputs[6]

                # image_metas as numpy array
                image_metas = image_metas.numpy()

                images_mc = np.repeat(images, self.__nb_mc, 0)

                # Wrap in variables
                images = Variable(images)
                rpn_match = Variable(rpn_match)
                rpn_bbox = Variable(rpn_bbox)
                gt_class_ids = Variable(gt_class_ids)
                gt_boxes = Variable(gt_boxes)
                gt_masks = Variable(gt_masks)

                # To GPU
                if self.__config.GPU_COUNT:
                    images = images.cuda()
                    rpn_match = rpn_match.cuda()
                    rpn_bbox = rpn_bbox.cuda()
                    gt_class_ids = gt_class_ids.cuda()
                    gt_boxes = gt_boxes.cuda()
                    gt_masks = gt_masks.cuda()

            
                mu_mcs = sess.run(self.__model.predictor,
                                      feed_dict={self.__model.x: x_mc, self.__model.keep_prob: 0.5})
                # Run detection
                results = model.detect([image])
                mu_mcs = np.asarray(mu_mcs, np.float32)[..., 0]
                h = sigmoid(np.mean(mu_mcs, 0))
                y = y[0, ..., 0]
                for t in thresh:
                    h_unc_thresh = self._clip_at_thresh(h, h, t)
                    stats = self._get_prd_stats(y, h_unc_thresh)
                    ustats[t]['fdr'] += stats['fdr']['all']
                    ustats[t]['tpr'] += stats['tpr']['all']
                    ustats[t]['dice'] += stats['dice']
                nb_subj += 1
                print('completed subject {}     {:.2f}m'.format(nb_subj, (timer() - start) / 60))
            for t in thresh:
                stats_writer.writerow(
                    [t, ustats[t]['fdr'] / nb_subj, ustats[t]['tpr'] / nb_subj, ustats[t]['dice'] / nb_subj])
        print("completed in {:.2f}m".format((timer() - start) / 60))

    def write_to_nrrd(self, out_dir):
        start = timer()
        nb_subj = 0
        print('-----------Starting predictions for each subject------------')
        for subj, tp, x, y in self.__data_gen:
            #x_mc = np.repeat(x, self.__nb_mc, 0)

            #mu_mcs, log_var_mcs = sess.run([self.__model.predictor, self.__model.log_variance],
            #                              feed_dict={self.__model.x: x_mc, self.__model.keep_prob: 0.5})

            mu_mcs = []
            log_var_mcs = []

            for i in range(0, self.__nb_mc):
                mu_temp, log_temp = sess.run([self.__model.predictor, self.__model.log_variance],
                                                feed_dict={self.__model.x: x, self.__model.keep_prob: 0.5})
                mu_mcs.append(mu_temp)
                log_var_mcs.append(log_temp)

            mu_mcs = np.asarray(mu_mcs)
            log_var_mcs = np.asarray(log_var_mcs)

            mu_mcs = mu_mcs[..., 0]
            mu_mcs = np.squeeze(mu_mcs, axis=1)    
            var_mcs = np.var(sigmoid(mu_mcs), 0)
                
            log_var_mcs = log_var_mcs[..., 0]
            log_var_mcs = np.squeeze(log_var_mcs, axis=1)
            mi = mi_uncertainty(sigmoid(mu_mcs))
            ent = entropy(sigmoid(mu_mcs))
            prd_var = prd_variance(log_var_mcs)
               
            h = sigmoid(np.mean(mu_mcs, 0))
            h = self._clip_at_thresh(h, h, thresh=0.00001)
            #y = y[0, ..., 0]
            #t2 = x[0, ..., 1]
       
            #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncmcvar.nrrd'), var_mcs, options=_OPTS) # unc measure, variance of mc samples
            #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_t2.nrrd'), t2, options=_OPTS) # t2 mri on its own
            # nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_h.nrrd'), h, options=_OPTS) # don't use this
            #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_target.nrrd'), y, options=_OPTS) # 'target' ground truth lesion labels
            #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncmi.nrrd'), mi, options=_OPTS) # mutual information unc. measure
            #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncent.nrrd'), ent, options=_OPTS) # entropy uncertainty measure
            #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncprvar.nrrd'), prd_var, options=_OPTS) # predicted variance unc. measure (2nd output of the model)

            h_mu_mcs = np.mean(sigmoid(mu_mcs), 0)
            #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_netseg.nrrd'), h, options=_OPTS) # network's segmentation ( y_hat). Use this one!!
            
            # don't use this
            # mu_nomcs = sess.run(self.__model.predictor, feed_dict={self.__model.x: x, self.__model.keep_prob: 1.0})
            # h_nomcs = sigmoid(mu_nomcs)[0, ..., 0]
            # nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_h_nomcs.nrrd'), h_nomcs, options=_OPTS)

            nb_subj += 1
            if nb_subj % 20 == 0:
                print('completed subject {}     {:.2f}m'.format(nb_subj, (timer() - start) / 60))
        print("completed in {:.2f}m".format((timer() - start) / 60))

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

    def global_dice(h, t):
        h = h.flatten()
        t = t.flatten()
        intersection = np.sum(h * t)
        union = np.sum(h) + np.sum(t)
        dice = (2. * intersection + _smooth) / (union + _smooth)
        return dice

    def sigmoid(x):
        _MIN_CLIP = -6
        _MAX_CLIP = 6
        x = np.asarray(x, dtype=np.float64)
        np.clip(x, _MIN_CLIP, _MAX_CLIP)
        return np.exp(x) / (1+np.exp(x))