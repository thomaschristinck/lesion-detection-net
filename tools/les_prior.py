import numpy as np
import nrrd
import os
from os import makedirs
from os.path import join
from argparse import ArgumentParser
'''
Takes a folder full of .nrrd files and averages the lesion masks - outputs a "lesion prior".
Lesion masks used in /usr/local/data/thomasc/unet_out/all_img are linearly co-registered
(MSLAQ). 
TO DO - make a lesion prior with masks from non-linearly co-registered MSLAQ data (also 
registered to patients) 
'''

def main(args):
	in_dir = args.input
	out_dir = args.output
	makedirs(out_dir, exist_ok=True)
	
	#Only use training set for lesion prior
	subjects = np.load(in_dir)
	subjects = [i[4:] for i in subjects]
	nb_folds = 10
	fold_length = len(subjects) // nb_folds
	train_idx = (nb_folds - 2) * fold_length    
	subjects = subjects[:train_idx]
	img_sum = 0
	img_count = 0

	for file in os.listdir(in_dir):
		if file.endswith("target.nrrd"):
			img, opts = nrrd.read(join(in_dir,file))
			img_sum = np.add(img_sum,img)
			img_count += 1
	print(img_count)
	les_prior = img_sum / img_count

	_OPTS = {'space': 'RAS', 'space directions': [(1, 0, 0), (0, 1, 0), (0, 0, 3)]}
	nrrd.write(join(out_dir, 'lesion_prior.nrrd'), les_prior, options=_OPTS)

def _parser():
	#Input path as /usr/local/data/tnair/thesis/data/mslaq/mslaq_subj_list.npy (blacktusk)
    usage = 'python les_prior.py -i input/path -o /output/path'
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Target Lesion Mask', required=True)
    parser.add_argument('-o', '--output', help='Lesion Prior Output', required=True)
    return parser

if __name__ == '__main__':
	main(_parser().parse_args())