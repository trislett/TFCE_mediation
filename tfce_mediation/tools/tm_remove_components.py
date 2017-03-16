#!/usr/bin/env python

#    tm_remove_components: remove noise components after ica
#    Copyright (C) 2016 Tristram Lett

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import argparse as ap
import pickle

from tfce_mediation.pyfunc import loadnifti,loadmgh,savenifti,savemgh

DESCRIPTION = "Remove noise components from ICA and rebuild image. Run tm_maths --fastica --timeplots first."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	datatype = ap.add_mutually_exclusive_group(required=True)
	datatype.add_argument("--voxel", 
		help="Voxel input",
		action="store_true")
	datatype.add_argument("--vertex", 
		help="Vertex input",
		action="store_true")
	datatype.add_argument("--bothhemi", 
		help="Special case in which vertex images from both hemispheres are input and processed together.",
		action="store_true")
	ap.add_argument("-i", "--input", 
		help="Text file of compoments to remove (as a single column).",
		nargs=1, 
		metavar=('*[.txt or .csv]'), 
		required=True)
	ap.add_argument("-o", "--output", 
		help="[Output Image Basename]",
		nargs=1, 
		required=True)
	ap.add_argument("--clean", 
		help="Remove the ICA_temp directory.", 
		action='store_true')
	return ap

def run(opts):
#check if ICA has been run
	if not os.path.exists("ICA_temp"):
		print "ICA_temp not found"
		exit()
#load mask
	if opts.bothhemi:
		lh_mask , lh_mask_data = loadmgh('ICA_temp/lh_mask.mgh')
		rh_mask , rh_mask_data = loadmgh('ICA_temp/rh_mask.mgh')
		lh_mask_index = (lh_mask_data != 0)
		rh_mask_index = (rh_mask_data != 0)
		midpoint = lh_mask_data[lh_mask_data==1].shape[0]
	else:
		if opts.voxel:
			mask , mask_data = loadnifti('ICA_temp/mask.nii.gz')
		if opts.vertex:
			mask , mask_data = loadmgh('ICA_temp/mask.mgh')
		mask_index = mask_data>.99

#load data
	rmcomps = np.genfromtxt(opts.input[0],delimiter=",",dtype='int')
	dump_ica = pickle.load( open( "ICA_temp/icasave.p", "rb" ) )
	fitcomps = np.load('ICA_temp/signals.npy').T
	selected=rmcomps-1
#zero out unwanted components
	fitcomps[:,selected] = 0
#rebuild image
	X_rec = dump_ica.inverse_transform(fitcomps)

#save image
	outname=opts.output[0]
	outname=outname.split('.gz',1)[0]
	outname=outname.split('.nii',1)[0]
	outname=outname.split('.mgh',1)[0]

	if opts.voxel:
		savenifti(X_rec, mask, mask_index, '%s.nii.gz' % outname)
	if opts.vertex:
		savemgh(X_rec, mask, mask_index, '%s.mgh' % outname)

	if opts.bothhemi:
		savemgh(X_rec[:midpoint], lh_mask, lh_mask_index, 'lh.%s.mgh' % outname)
		savemgh(X_rec[midpoint:], rh_mask, rh_mask_index, 'rh.%s.mgh' % outname)

	if opts.clean:
		os.system("rm -rf ICA_temp")

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)

