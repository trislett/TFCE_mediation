#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
import math

def find_nearest(array,value,p_array):
	idx = np.searchsorted(array, value, side="left")
	if idx == len(p_array):
		return p_array[idx-1]
	elif math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
		return p_array[idx-1]
	else:
		return p_array[idx]

if len(sys.argv) < 3:
	print "Usage: %s [tfce_img] [perm_tfce_max]" % (str(sys.argv[0]))
else:
	cmdargs = str(sys.argv)
	arg_tfce_img = str(sys.argv[1])
	arg_perm_tfce_max = str(sys.argv[2])
	perm_tfce_max = np.genfromtxt(arg_perm_tfce_max, delimiter=',')

	tfce_img = nib.load(arg_tfce_img)
	data_tfce_img = tfce_img.get_data()
	affine_tfce_img = tfce_img.get_affine()
	header_tfce_img = tfce_img.get_header()
	corrp_img = np.zeros(tfce_img.shape)

	sorted_perm_tfce_max=np.sort(perm_tfce_max)
	p_array=np.zeros(perm_tfce_max.shape)
	num_perm=perm_tfce_max.shape[0]
	for j in xrange(num_perm):
		p_array[j] = np.true_divide(j,num_perm)
	thresh = np.array(data_tfce_img > 0)
	ind=np.where(thresh)
	for x,y,z in zip(ind[0],ind[1],ind[2]):
		corrp_img[x,y,z] = find_nearest(sorted_perm_tfce_max,data_tfce_img[x,y,z],p_array)
	outputdir = os.path.dirname(arg_tfce_img)
	temp_outname = os.path.basename(arg_tfce_img)
	temp_outname, _ = os.path.splitext(temp_outname)
	temp_outname, _ = os.path.splitext(temp_outname)
	tfce_fweP_name = "%s_FWEcorrP.nii.gz" % (temp_outname)
	nib.save(nib.Nifti1Image(corrp_img,affine_tfce_img),tfce_fweP_name)
