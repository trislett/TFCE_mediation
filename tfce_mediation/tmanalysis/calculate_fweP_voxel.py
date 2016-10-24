#!/usr/bin/python

import os
import numpy as np
import nibabel as nib
import argparse as ap
import math

DESCRIPTION = "Calculate 1-P[FWE] 3D image from max TFCE values from randomisation."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i", "--input", 
		nargs=2,
		metavar=('*.nii.gz', '*.csv'),
		help="[tfce_image] [perm_tfce_max]",
		required=True)
	return ap

#find nearest permuted TFCE max value that corresponse to family-wise error rate 
def find_nearest(array,value,p_array):
	idx = np.searchsorted(array, value, side="left")
	if idx == len(p_array):
		return p_array[idx-1]
	elif math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
		return p_array[idx-1]
	else:
		return p_array[idx]

def run(opts):
	arg_tfce_img = str(opts.input[0])
	arg_perm_tfce_max = str(opts.input[1])
	perm_tfce_max = np.genfromtxt(arg_perm_tfce_max, delimiter=',')

#load data
	tfce_img = nib.load(arg_tfce_img)
	data_tfce_img = tfce_img.get_data()
	affine_tfce_img = tfce_img.get_affine()
	corrp_img = np.zeros(tfce_img.shape)

#sort max tfce values
	sorted_perm_tfce_max=np.sort(perm_tfce_max)
	p_array=np.zeros(perm_tfce_max.shape)
	num_perm=perm_tfce_max.shape[0]
	for j in xrange(num_perm):
		p_array[j] = np.true_divide(j,num_perm)
	thresh = np.array(data_tfce_img > 0)
	ind=np.where(thresh)
	for x,y,z in zip(ind[0],ind[1],ind[2]):
		corrp_img[x,y,z] = find_nearest(sorted_perm_tfce_max,data_tfce_img[x,y,z],p_array)

#output corrected image,and printout accuracy based on number of permuations
	temp_outname = os.path.basename(arg_tfce_img)
	temp_outname, _ = os.path.splitext(temp_outname)
	temp_outname, _ = os.path.splitext(temp_outname)
	tfce_fweP_name = "%s_FWEcorrP.nii.gz" % (temp_outname)
	print "The accuracy is p = 0.05 +/- %.4f" % (2*(np.sqrt(0.05*0.95/num_perm)))
	nib.save(nib.Nifti1Image(corrp_img,affine_tfce_img),tfce_fweP_name)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
