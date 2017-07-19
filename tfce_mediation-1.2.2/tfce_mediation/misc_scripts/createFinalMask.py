#!/usr/bin/python

import numpy as np
import nibabel as nib
import argparse as ap

ap = ap.ArgumentParser(description="Correct mean or skeleton mask via T1toStd masks for all subjects")
ap.add_argument("-i", "--FAMask", nargs=1, help="[FAMask]", metavar=('*.nii.gz'), default=['mean_FA_skeleton_mask.nii.gz'])
ap.add_argument("-m", "--masks", nargs='+', help="[T1toStdMasks] ...", metavar=('*.nii.gz'), required=True)
opts = ap.parse_args()
numMerge=len(opts.masks)

skeletonMask = nib.load(opts.FAMask[0]) 
skeletonMaskData = skeletonMask.get_data()
affine = skeletonMask.get_affine()
header = skeletonMask.get_header()
outMask = skeletonMask.get_data()
data_index = skeletonMaskData>0.99
for i in xrange(numMerge):
	outMask[data_index]=np.multiply(outMask[data_index],nib.load(opts.masks[i]).get_data()[data_index])
nib.save(nib.Nifti1Image(outMask.astype(np.float32, order = "C"),affine),opts.FAMask[0])

