#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import argparse as ap

DESCRIPTION = "Extract mean values from a label file (i.e. a cluster)"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i", "--input", 
		help="[surface] (e.g., lh.all.area.00.mgh)", 
		nargs=1, 
		metavar=('*.mgh'), 
		required=True)
	ap.add_argument("-l", "--label", 
		help="[label file]", 
		nargs=1, 
		metavar=('*.label-????'), 
		required=True)
	return ap

def run(opts):
	arg_allsurf = opts.input[0]
	label = opts.label[0]
	img_allsurf = nib.freesurfer.mghformat.load(arg_allsurf)
	alldata_full = img_allsurf.get_data()
	vertices = np.genfromtxt(label, delimiter='  ', skip_header=2, usecols= (0), dtype='int32')
	values = np.zeros((len(vertices),alldata_full.shape[3]))
	ite=0
	for i in vertices:
		values[ite,:]=alldata_full[int(i),0,0,:]
 		ite+=1
	meanvalues=np.sum(values,axis=0)/len(vertices)
	np.savetxt(("%s.mean.txt" % label), meanvalues, delimiter=',',fmt='%1.5f')

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
