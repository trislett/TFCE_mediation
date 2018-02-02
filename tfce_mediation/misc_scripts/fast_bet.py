#!/usr/bin/python

import os
import numpy as np
import nibabel as nib
import argparse as ap
from skimage import filters

DESCRIPTION = "Brain extraction using autothresholding and FSL BET"

# various methods for choosing thresholds automatically
def autothreshold(data, threshold_type = 'otsu', z = 2.3264, set_max_ceiling = True):
	if threshold_type.endswith('_p'):
		data = data[data>0]
	else:
		data = data[data!=0]
	if (threshold_type == 'otsu') or (threshold_type == 'otsu_p'):
		lthres = filters.threshold_otsu(data)
		uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		# Otsu N (1979) A threshold selection method from gray-level histograms. IEEE Trans. Sys., Man., Cyber. 9: 62-66.
	elif (threshold_type == 'li')  or (threshold_type == 'li_p'):
		lthres = filters.threshold_li(data)
		uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		# Li C.H. and Lee C.K. (1993) Minimum Cross Entropy Thresholding Pattern Recognition, 26(4): 617-625
	elif (threshold_type == 'yen') or (threshold_type == 'yen_p'):
		lthres = filters.threshold_yen(data)
		uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		# Yen J.C., Chang F.J., and Chang S. (1995) A New Criterion for Automatic Multilevel Thresholding IEEE Trans. on Image Processing, 4(3): 370-378.
	elif threshold_type == 'zscore_p':
		lthres = data.mean() - (z*data.std())
		uthres = data.mean() + (z*data.std())
		if lthres < 0:
			lthres = 0.001
	else:
		lthres = data.mean() - (z*data.std())
		uthres = data.mean() + (z*data.std())
	if set_max_ceiling: # for the rare case when uthres is larger than the max value
		if uthres > data.max():
			uthres = data.max()
	return lthres, uthres

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i", "--image",
		nargs=1,
		help="Input a image (Nifti, MGH, MINC)", 
		metavar=('*'),
		required = True)
	ap.add_argument("-n4", "--AntsN4correction",
		action='store_true',
		help="Perform ANTS N4 Bias correction.")
	ap.add_argument("-t", "--thresholdalgorithm",
		help = "Method used to set the the lower threshold if thresholds are not supplied (Default is otsu).",
		choices = ['otsu', 'otsu_p', 'li', 'li_p', 'yen', 'yen_p', 'zscore', 'zscore_p'])
	ap.add_argument("-o", "--output",
		nargs=1,
		help="Set output")
	ap.add_argument("-r", "--replace",
		nargs=1,
		help="Replaces a image file (creates a backup)")
	return ap

def run(opts):

	img = nib.as_closest_canonical(nib.load(opts.image[0])) # for pesky LR flipping
	data = img.get_data()
	hdr = img.get_header()
	low_threshold, _ = autothreshold(data, threshold_type = opts.thresholdalgorithm)
	print(low_threshold)
	mask = np.zeros_like(data)
	mask[:] = data
	mask[mask < low_threshold] = 0
#	mask[mask != 0] = 1
	nib.save(nib.Nifti1Image(mask.astype(np.float32, order = "C"),affine=img.affine),'temp.nii.gz')
	os.system(os.environ["FSLDIR"] + "/bin/bet temp.nii.gz temp_brain.nii.gz -m -f 0.3")
	betmask = nib.as_closest_canonical(nib.load('temp_brain_mask.nii.gz')).get_data()
	data[betmask!=1] = 0
	if opts.output:
		nib.save(nib.Nifti1Image(data.astype(np.float32, order = "C"),affine=img.affine), opts.output[0])
	elif opts.replace:
		base, name = os.path.split(opts.replace[0])
		os.system("mv %s %s/backup_%s" % (opts.replace[0], base, name))
		nib.save(nib.Nifti1Image(data.astype(np.float32, order = "C"),affine=img.affine), opts.replace[0])
	else:
		nib.save(nib.Nifti1Image(data.astype(np.float32, order = "C"),affine=img.affine),'bet_' + opts.image[0])
	os.system("rm temp*.nii.gz")



if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
