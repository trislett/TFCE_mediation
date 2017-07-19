#!/usr/bin/env python

import os
import argparse as ap

DESCRIPTION = "Python wrapper to run FSL's cluster_results on pFWE corrected statistics voxel images. The default threshold is 0.95 (i.e., pFWE<0.05)"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i", "--image", 
		help="stat_image.nii.gz", 
		nargs=1, 
		metavar=('*.nii.gz'), 
		required=True)
	ap.add_argument("-t", "--threshold", 
		help="1-P(FWE) threshold (default is 0.95)", 
		default=[0.95], 
		nargs=1)
	return ap

def run(opts):

	statimage=str(opts.image[0])
	thresh=float(opts.threshold[0])

	os.system('mkdir -p cluster_results; $FSLDIR/bin/cluster -i %s -t %1.2f --mm --scalarname="1-p" -o cluster_results/$(basename %s .nii.gz)_clusters > cluster_results/$(basename %s .nii.gz)_results' % (statimage,thresh,statimage,statimage))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
