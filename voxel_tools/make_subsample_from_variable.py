#!/usr/bin/python

import sys
import numpy as np

table = np.load('python_temp/raw_nonzero.npy')
imgdata = table.T
header_mask = np.load('python_temp/header_mask.npy')
affine_mask = np.load('python_temp/affine_mask.npy')
num_voxel = np.load('python_temp/num_voxel.npy')
num_subjects = np.load('python_temp/num_subjects.npy')

if len(sys.argv) < 2:
	print "*******************************************************************"
	print "Usage: %s [1D Subgrouping Variable]" % (str(sys.argv[0]))
	print "  "
	print "Careful. Creates a subgroub based on missing data from an analysis."
	print "The subgrouping variable should be a 1D text file list with missing"
	print "variables coded as a string (e.g. NA or NaN)."
	print "--- You must be in same directory as python_temp ---"
	print "*******************************************************************"
else:

	cmdargs = str(sys.argv)
	arg_subgroupvariable = str(sys.argv[1])
	subgroupvariable = np.genfromtxt(arg_subgroupvariable, delimiter=',')
	masking_variable=np.isfinite(subgroupvariable)
	subdata=imgdata[masking_variable]
	num_subjects_new = subdata.T.shape[1]
	np.save('python_temp/raw_nonzero',subdata.T)
	np.save('python_temp/raw_nonzero_orignal',table)
	np.save('python_temp/num_subjects',num_subjects_new)
	np.save('python_temp/num_subjects_orignal',num_subjects)
	np.save('python_temp/masking_variable',masking_variable)
	np.savetxt('masking_variable.out', masking_variable, delimiter=',',fmt='%i')
	print "run ./make_all_fa_skel_fromNumpy.py for the 4D nifti skeleton, and the skeleton mask."

