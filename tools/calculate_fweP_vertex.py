#!/usr/bin/python

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
	print "Usage: %s [tfce_img.mgh] [perm_tfce_max]]" % (str(sys.argv[0]))
else:

	cmdargs = str(sys.argv)
	arg_tfce_image = str(sys.argv[1])
	arg_maxTFCE = str(sys.argv[2])
	perm_tfce_max = np.genfromtxt(arg_maxTFCE, delimiter=",")
	img = nib.freesurfer.mghformat.load(arg_tfce_image)
	arg_tfce_image_noext = arg_tfce_image.split('.mgh',1)[0]
	data_full = img.get_data()
	data = np.squeeze(data_full)
	affine_mask = img.get_affine()
	bin_mask = data>0
	masked_data=data[bin_mask]

	num_perm=perm_tfce_max.shape[0]
	sorted_perm_tfce_max=np.sort(perm_tfce_max)
	p_array=np.zeros_like(sorted_perm_tfce_max)
	corrp_img = np.zeros(masked_data.shape)

	for j in xrange(num_perm):
		p_array[j] = np.true_divide(j,num_perm)
	cV=0
	for k in masked_data:
		corrp_img[cV] = find_nearest(sorted_perm_tfce_max,k,p_array)
		cV+=1
	outmask=np.zeros_like(data_full)
	outmask[bin_mask,0,0]=corrp_img
	nib.save(nib.Nifti1Image(outmask,affine_mask),"%s_FWEcorrP.mgh" % (arg_tfce_image_noext))
	print "The accuracy is p = 0.05 +/- %.4f" % (2*(np.sqrt(0.05*0.95/num_perm)))
