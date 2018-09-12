#!/usr/bin/env python

#    Load mean_FA_skeleton_mask and all_FA_skeletonised into tfce_mediation
#    Copyright (C) 2016  Tristram Lett

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import nibabel as nib
import argparse

from tfce_mediation.pyfunc import import_voxel_neuroimage


DESCRIPTION = "Initial step to load in a 4D NifTi or MINC volumetric image, and its corresponding mask (binary image)."

def getArgumentParser(parser = argparse.ArgumentParser(description = DESCRIPTION)):
	parser.add_argument("-i", "--input", 
		nargs=2, 
		help="[4D_image] [Mask] (default: %(default)s)", 
		metavar=('*.nii.gz or *.mnc', '*.nii.gz or *.mnc'), 
		default=['all_FA_skeletonised.nii.gz','mean_FA_skeleton_mask.nii.gz'])
	inputtype = parser.add_mutually_exclusive_group(required=False)
	inputtype.add_argument("-f", "--files", 
		nargs='+', 
		help="Neuroimage file location with wildcards. USAGE: -f {neuroimage} ...", 
		metavar=('*.nii.gz or *.mnc'))
	inputtype.add_argument("-l", "--filelist", 
		nargs=1, 
		help="Import neuroimages from text file. -l {1D text file}", 
		metavar=('*.csv'))
	parser.add_argument("-m", "--mask", 
		nargs=1, 
		help="Must be used with --files or --filelist. USAGE: -m {Mask}", 
		metavar=('*.nii.gz or *.mnc'))
	return parser

def run(opts):
	if opts.mask:
		mask_name = opts.mask[0]
	else:
		mask_name = opts.input[1]
	if not os.path.isfile(mask_name):
		print('Error: %s not found. Please use -i or -m option.' % mask_name)
		quit()
	img_mask = import_voxel_neuroimage(mask_name)
	data_mask = img_mask.get_data()
	affine_mask = img_mask.get_affine()
	header_mask = img_mask.get_header()
	mask_index = data_mask > 0.99

	if opts.files:
		nonzero_data = []
		for image_path in opts.files:
			nonzero_data.append(import_voxel_neuroimage(image_path, mask_index))
		nonzero_data = np.array(nonzero_data).T
	elif opts.filelist:
		nonzero_data = []
		for image_path in np.genfromtxt(opts.filelist[0], delimiter=',', dtype=None):
			nonzero_data.append(import_voxel_neuroimage(image_path, mask_index))
		nonzero_data = np.array(nonzero_data).T
	else:
		nonzero_data = import_voxel_neuroimage(opts.input[0], mask_index)

	# check mask
	mean = np.mean(nonzero_data, axis=1)
	if len(mean[mean==0]) != 0:
		print("Warning: the mask contains data that is all zeros in the 4D image. Creating a new mask:\t new_%s"% mask_name)
		new_mask = np.zeros_like(mean)
		new_mask[mean!=0] = 1
		data_mask[data_mask!=0] = new_mask[:]
		nib.save(nib.Nifti1Image(data_mask.astype(np.float32),affine=img_mask.affine), 'new_%s' % mask_name)
		nonzero_data = data_all[data_mask>0.99]

	if not os.path.exists('python_temp'):
		os.mkdir('python_temp')

	np.save('python_temp/raw_nonzero',nonzero_data.astype(np.float32, order = "C"))
	np.save('python_temp/header_mask',header_mask)
	np.save('python_temp/affine_mask',affine_mask)
	np.save('python_temp/data_mask',data_mask)
	num_voxel = nonzero_data.shape[0]
	num_subjects = nonzero_data.shape[1]
	np.save('python_temp/num_voxel',num_voxel)
	np.save('python_temp/num_subjects',num_subjects)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
