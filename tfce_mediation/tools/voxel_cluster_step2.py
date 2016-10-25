#!/usr/bin/env python

import numpy as np
import nibabel as nib
import argparse as ap

DESCRIPTION = "Extract mean voxel values from clusters after running cluster_step1. Probably best to run from cluster_results directory."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):

	ap.add_argument("-i", "--image", 
		help="Clusters Nifti image", 
		nargs=1, 
		required=True,
		metavar=('*_clusters.nii.gz'))
	ap.add_argument("-c", "--cluster", 
		help="Which clusters to extract from results file", 
		nargs='+', 
		type=int,
		metavar=('INT'))
	ap.add_argument("-d", "--dir",
		help="python_temp directory", 
		default=['../../python_temp'], 
		nargs=1)
	return ap

def run(opts):

	image_name = opts.image[0].split('.nii.gz',1)[0]

	raw_nonzero = np.load('%s/raw_nonzero.npy' % opts.dir[0])
	data_mask = np.load('%s/data_mask.npy'% opts.dir[0])
	num_subjects = np.load('%s/num_subjects.npy'% opts.dir[0])

	img_label = nib.load(opts.image[0])
	label_data = img_label.get_data()
	label_data = label_data * data_mask

	allFA4D = np.zeros((182, 218, 182, num_subjects))
	allFA4D[data_mask>0.99] = raw_nonzero

	for i in range(len(opts.cluster)):
		meanvalue=allFA4D[label_data==opts.cluster[i]].mean(axis=0)
		np.savetxt("%s_mean_clusterIndex%s.csv" % (image_name,opts.cluster[i]), meanvalue, delimiter=",")

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
