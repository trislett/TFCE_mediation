#!/usr/bin/env python

import os
import numpy as np
import argparse as ap

DESCRIPTION = "Creates a subgroub based on missing data from an analysis. The subgrouping variable should be a 1D text file list with missing variables coded as a string (e.g. NA or NaN)."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	ap.add_argument("-i", "--input", 
		help="[1D Subgrouping Variable]", 
		nargs=1, 
		metavar=('*.csv'))
	return ap

def run(opts):

	if not os.path.exists("python_temp"):
		print "python_temp is missing"

	arg_subgroupvariable = str(opts.input[0])
#load data
	table = np.load('python_temp/raw_nonzero.npy')
	imgdata = table.T
	header_mask = np.load('python_temp/header_mask.npy')
	affine_mask = np.load('python_temp/affine_mask.npy')
	num_voxel = np.load('python_temp/num_voxel.npy')
	num_subjects = np.load('python_temp/num_subjects.npy')

#make subset
	subgroupvariable = np.genfromtxt(arg_subgroupvariable, delimiter=',')
	masking_variable=np.isfinite(subgroupvariable)
	subdata=imgdata[masking_variable]
	num_subjects_new = subdata.T.shape[1]
#write data
	np.save('python_temp/raw_nonzero',subdata.T)
	np.save('python_temp/raw_nonzero_orignal',table)
	np.save('python_temp/num_subjects',num_subjects_new)
	np.save('python_temp/num_subjects_orignal',num_subjects)
	np.save('python_temp/masking_variable',masking_variable)
	np.savetxt('masking_variable.out', masking_variable, delimiter=',',fmt='%i')

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
