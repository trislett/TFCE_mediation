#!/usr/bin/env python

import os
import argparse as ap

DESCRIPTION = "Python wrapper to run mri_surfcluster on PFWE corrected statistics surface images. The default threshold is 0.95 (i.e., pFWE<0.05)"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i", "--image", 
		help="stat_image_?h.mgh", 
		nargs=1, 
		metavar=('*.mgh'), 
		required=True)
	ap.add_argument("--hemi", help="Hemisphere", 
		choices=['lh', 'rh'],
		required=True)
	ap.add_argument("-t", "--threshold", 
		help="1-P(FWE) threshold (default is 0.95)", 
		default=[0.95],
		nargs=1)
	return ap

def run(opts):

	statimage=str(opts.image[0])
	hemi=str(opts.hemi)
	thresh=float(opts.threshold[0])

	os.system("mkdir -p cluster_results; $FREESURFER_HOME/bin/mri_surfcluster --in %s --thmin %1.2f --hemi %s --subject fsaverage --o cluster_results/$(basename %s .mgh)_maskedvalues.mgh --olab cluster_results/$(basename %s .mgh).label --ocn cluster_results/$(basename %s .mgh)_label_surface.mgh > cluster_results/$(basename %s .mgh).output" % (statimage,thresh,hemi,statimage,statimage,statimage,statimage))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
