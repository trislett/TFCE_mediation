#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import argparse as ap
from sklearn.decomposition import PCA

# Detects outliers using median absolute deviation
# Reference
# Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
# Handle Outliers", The ASQC Basic References in Quality Control:
# Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
def mad_based_outlier(points, thresh=3.5):
	if len(points.shape) == 1:
		points = points[:,None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)
	modified_z_score = 0.6745 * diff / med_abs_deviation
	return modified_z_score > thresh


DESCRIPTION = "Calculate mean FA values from label after running STEP_0."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-l", "--label", 
		help="Default is JHU-ICBM-labels-1mm.nii.gz", 
		default=['%s/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz' % os.environ.get('FSLDIR')], 
		nargs=1)
	ap.add_argument("-r", "--range",
		help="Label range. Default is 1 48", 
		type=int, 
		default=[1,48], 
		nargs=2, 
		metavar=('INT', 'INT'))
	ap.add_argument("-m", "--mask", 
		help="Specify a mask", 
		nargs=1)
	return ap

def run(opts):
	print("Calculating mean values.")

	if not os.path.exists("python_temp"):
		print("python_temp is missing ya dingus!")
		quit()

	raw_nonzero = np.load('python_temp/raw_nonzero.npy')
	data_mask = np.load('python_temp/data_mask.npy')
	num_subjects = np.load('python_temp/num_subjects.npy')


	if opts.mask:
		img_label = nib.load(opts.mask[0])
		label_data = img_label.get_data()
		label_nonzero = label_data[data_mask!=0]
		np.savetxt("Mean_voxel_mask.csv", allFA4D[label_nonzero==1].mean(axis=0), delimiter=",")
	else:
		img_label = nib.load(opts.label[0])
		label_data = img_label.get_data()
		label_nonzero = label_data[data_mask!=0]
		start=opts.range[0]
		stop=opts.range[1] + 1
		meanvalue = np.zeros((num_subjects,int(stop-start)))
		outliers = np.zeros((num_subjects,int(stop-start)))
		for i in range(start,stop):
			if not len (raw_nonzero[label_nonzero==i])==0:
				mean = raw_nonzero[label_nonzero==i].mean(axis=0)
				meanvalue[:,(i-1)] = mean
				outliers[:,(i-1)] = mad_based_outlier(mean)*1
				print("LABEL %s:\t%d\tvoxels" % (i,len(label_data[label_data==i])))
			else:
				print("Error: Label %d contains zero voxels." % i)
		np.savetxt("Mean_voxel_label.csv", meanvalue, delimiter=",")
		np.savetxt("Outlier_voxel_label.csv", outliers, delimiter=",")


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
