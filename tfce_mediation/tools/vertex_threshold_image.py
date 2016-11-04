#!/usr/bin/env python

import os
import sys
import numpy as np
import nibabel as nib
import argparse as ap

DESCRIPTION = "Create a mask at a specific threshold."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):

	ap.add_argument("-i", "--input",  
		nargs=1, 
		metavar=('*.mgh'), 
		required=True)
	ap.add_argument("-t", "--threshold",  
		nargs=1, 
		required=True)
	ap.add_argument("-n", "--neg", 
		help='output negative threshold image', 
		required=False, 
		action="store_true")
	return ap

def run(opts):
	arg_surf = str(opts.input[0])
	thresh = float(opts.threshold[0])
	img_surf = nib.freesurfer.mghformat.load(arg_surf)
	data_full = img_surf.get_data()
	mask = data_full>thresh
	mask = np.array(mask*1).astype('>f4')
	nib.save(nib.freesurfer.mghformat.MGHImage(mask,img_surf.get_affine()),'bin_mask_%1.1f_%s' % (thresh,arg_surf))
	if opts.neg:
		data_full = data_full*-1
		mask = data_full>thresh
		mask = np.array(mask*1).astype('>f4')
		nib.save(nib.freesurfer.mghformat.MGHImage(mask,img_surf.get_affine()),'bin_mask_%1.1f_neg_%s' % (thresh,arg_surf))


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
