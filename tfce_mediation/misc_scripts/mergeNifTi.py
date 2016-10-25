#!/usr/bin/python

import numpy as np
import nibabel as nib
import argparse as ap

ap = ap.ArgumentParser(description="Fast merging for nifti images")
ap.add_argument("-o", "--output", nargs=1, help="[4D_image]", metavar=('*.nii.gz'), required=True)
ap.add_argument("-i", "--input", nargs='+', help="[3Dimage] ...", metavar=('*.nii.gz'), required=True)
ap.add_argument("-m", "--mask", nargs=1, help="[3Dimage]", metavar=('*.nii.gz'))
opts = ap.parse_args()
outname=opts.output[0]
numMerge=len(opts.input)


firstVol = nib.load(opts.input[0])
firstVolData = firstVol.get_data()
affine = firstVol.get_affine()
header = firstVol.get_header()
allVolume = np.zeros((firstVolData.shape[0], firstVolData.shape[1], firstVolData.shape[2], numMerge)).astype(np.float32, order = "C")

if opts.mask:
	img_mask = nib.load(opts.mask[0])
	data_mask = img_mask.get_data()
	data_index = data_mask>0.99
	allMaskVolume = allVolume[data_index]
	for i in xrange(numMerge):
		print "merging image %s" % opts.input[i]
		allMaskVolume[:,i] = nib.load(opts.input[i]).get_data()[data_index]
	allVolume[data_index]=allMaskVolume
else:
	for i in xrange(numMerge):
		print "merging image %s" % opts.input[i]
		allVolume[:,:,:,i] = nib.load(opts.input[i]).get_data()

nib.save(nib.Nifti1Image(allVolume.astype(np.float32, order = "C"),affine),outname)
