import nrrd
from os.path import join
import numpy as np
import os
import h5py
from timeit import default_timer as timer
from argparse import ArgumentParser

def main(args):
	in_dir = '/usr/local/data/thomasc/unet_out/all_img/'
	f = h5py.File(join(args.output, "det_data.h5py"), "w")
	file_paths = os.listdir(in_dir)
	nb_subjects = 0
	'''
	Saves data with structure Subject_id/Time_point/Modality
	'''
	start = timer()
	for file in sorted(file_paths):
		img, opts = nrrd.read(join(in_dir, file))
		subj_id = '_'.join(file.split('_')[0:2])
		tp = file.split('_')[2]
		mod = file.split('_')[3].split('.')[0]
	
		if subj_id not in list(f.keys()):
			f.create_group(subj_id)
			nb_subjects += 1
			if nb_subjects % 50 == 0:
				print('Completed {} subjects     {:.2f}m'.format(nb_subjects, (timer() - start) / 60))
		
		if tp not in list(f[subj_id].keys()):
			f[subj_id].create_group(tp)

		if mod not in list(f[subj_id][tp].keys()):
			f[subj_id][tp].create_dataset(mod, data=img, compression="gzip")

	f.close()

def _parser():
    usage = 'python hdf5_converter.py -o /output/path'
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', help='hdf5 Output', required=True)
    return parser

if __name__ == '__main__':
	main(_parser().parse_args())

