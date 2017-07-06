#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import argparse as ap

DESCRIPTION = "Convert numpy voxel data (from python_temp) to 4D nifti image"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	ap.add_argument("-o", "--out", 
		help="Specify output name for 4D volume. Default: %(default)s).", 
		nargs=1, 
		metavar=('*.nii.gz'),
		default=['all_FA_skeletonised.nii.gz'])
	ap.add_argument("-n", "--specifiyrawdata", 
		help="Specify which *.npy data to convert. Default: %(default)s).", 
		nargs=1, 
		metavar=('*.npy'),
		default=['python_temp/raw_nonzero.npy'])
	return ap

def run(opts):
	outname=str(opts.out[0])
	inname=str(opts.specifiyrawdata[0])
#load numpy data
	all_data = np.load(inname)
	header_mask = np.load('python_temp/header_mask.npy')
	affine_mask = np.load('python_temp/affine_mask.npy')
	data_mask = np.load('python_temp/data_mask.npy')
	num_voxel = np.load('python_temp/num_voxel.npy')
	num_subjects = np.load('python_temp/num_subjects.npy')
	datadim=np.array(data_mask.shape)

	out_4d = np.zeros((datadim[0],datadim[1],datadim[2],num_subjects))
	out_4d[data_mask>0.99]=all_data

	if os.path.isfile(outname) == True:
		os.system("mv %s backup_%s" % (outname,outname))
		print "Warning. Any previous backups would have been over-written"
#write nifti data
	nib.save(nib.Nifti1Image(out_4d.astype(np.float32, order = "C"),affine_mask),(outname))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
