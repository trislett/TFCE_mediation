#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
# import csv
import statsmodels.api as sm

table = np.load('python_temp/raw_nonzero.npy')
header_mask = np.load('python_temp/header_mask.npy')
affine_mask = np.load('python_temp/affine_mask.npy')
data_mask = np.load('python_temp/data_mask.npy')
num_voxel = np.load('python_temp/num_voxel.npy')
num_subjects = np.load('python_temp/num_subjects.npy')

cmdargs = str(sys.argv)
arg_covars = str(sys.argv[1])
covars = np.genfromtxt(arg_covars, delimiter=',')
table_agecorr = table
x_covars = sm.add_constant(covars)
a_c = np.linalg.lstsq(x_covars, table.T)[0]
resids = table.T - np.dot(x_covars,a_c)
table_agecorr = resids.T

if not os.path.exists('debug'):
	os.mkdir('debug')
os.chdir('debug')

out_4d = np.zeros((data_mask.shape[0], data_mask.shape[1], data_mask.shape[2],num_subjects))
out_4d[data_mask>0.99]=table_agecorr

nib.save(nib.Nifti1Image(out_4d,affine_mask),'all_FA_skeletonised_resids.nii.gz')

