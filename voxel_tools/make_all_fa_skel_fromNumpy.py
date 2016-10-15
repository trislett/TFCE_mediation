#!/usr/bin/python

import os
import numpy as np
import nibabel as nib

table = np.load('python_temp/raw_nonzero.npy')
header_mask = np.load('python_temp/header_mask.npy')
affine_mask = np.load('python_temp/affine_mask.npy')
data_mask = np.load('python_temp/data_mask.npy')
num_voxel = np.load('python_temp/num_voxel.npy')
num_subjects = np.load('python_temp/num_subjects.npy')

out_4d = np.zeros((182, 218, 182,num_subjects))
out_4d[data_mask>0.99]=table

if os.path.isfile('all_FA_skeletonised.nii.gz') == True:
	os.system("mv all_FA_skeletonised.nii.gz backup_all_FA_skeletonised.nii.gz; mv mean_FA_skeleton.nii.gz backup_mean_FA_skeleton.nii.gz")
	print "Warning. Any previous backups would have been over-written"

nib.save(nib.Nifti1Image(out_4d,affine_mask),'all_FA_skeletonised.nii.gz')
nib.save(nib.Nifti1Image(data_mask,affine_mask),'mean_FA_skeleton_mask.nii.gz')
