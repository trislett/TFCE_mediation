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

DESCRIPTION = "Initial step to load in a 4D NifTi or MINC volumetric image, and its corresponding mask (binary image)."

def getArgumentParser(parser = argparse.ArgumentParser(description = DESCRIPTION)):
	parser.add_argument("-i", "--input", 
		nargs=2, 
		help="[4D_image] [Mask] (default: %(default)s)", 
		metavar=('*.nii.gz or *.mnc', '*.nii.gz or *.mnc'), 
		default=['all_FA_skeletonised.nii.gz','mean_FA_skeleton_mask.nii.gz'])
	return parser

def run(opts):
	mask_name = opts.input[1]
	if not os.path.isfile(mask_name):
		print 'Error %s not found. Please use -i option.' % mask_name
		quit()
	img_mask = nib.load(mask_name)
	data_mask = img_mask.get_data()
	affine_mask = img_mask.get_affine()
	header_mask = img_mask.get_header()

	#check if minc file
	img_all_name = opts.input[0]
	_, file_ext = os.path.splitext(img_all_name)
	if file_ext == '.gz':
		_, file_ext = os.path.splitext(img_all_name)
		if file_ext == '.mnc':
			imgext = '.mnc'
			img_all = nib.load(img_all_name)
		else:
			imgext = '.nii.gz'
			os.system("zcat %s > temp_4d.nii" % img_all_name)
			img_all = nib.load('temp_4d.nii')
	elif file_ext == '.nii':
		imgext = '.nii.gz' # default to zipped images
		img_all = nib.load(img_all_name)
	else:
		print 'Error filetype for %s is not supported' % img_all_name
		quit()

	data_all = img_all.get_data()

	nonzero_data = data_all[data_mask>0.99]

	if not os.path.exists('python_temp'):
		os.mkdir('python_temp')

	np.save('python_temp/raw_nonzero',nonzero_data.astype(np.float32, order = "C"))
	np.save('python_temp/header_mask',header_mask)
	np.save('python_temp/affine_mask',affine_mask)
	np.save('python_temp/data_mask',data_mask)
	np.save('python_temp/imgext',imgext)
	num_voxel = nonzero_data.shape[0]
	num_subjects = nonzero_data.shape[1]
	np.save('python_temp/num_voxel',num_voxel)
	np.save('python_temp/num_subjects',num_subjects)
	os.system("rm temp_4d.nii")

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
