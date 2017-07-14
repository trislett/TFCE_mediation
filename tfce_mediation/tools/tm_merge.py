#!/usr/bin/env python

#    tm_merge: efficiently merge *mgh or *nii.gz files
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
import nibabel as nib
import argparse as ap
from tfce_mediation.pyfunc import *
from tfce_mediation.pyfunc import zscaler
from sklearn.decomposition import FastICA

DESCRIPTION = "Efficient merging for Nifti or MGH images"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	datatype = ap.add_mutually_exclusive_group(required=True)
	datatype.add_argument("--voxel", 
		help="Voxel input",
		action="store_true")
	datatype.add_argument("--vertex", 
		help="Vertex input",
		action="store_true")
	ap.add_argument("-o", "--output", nargs=1, help="[4D_image]", metavar=('*.nii.gz or *.mgh'), required=True)
	ap.add_argument("-i", "--input", nargs='+', help="[3Dimage] ...", metavar=('*.nii.gz or *.mgh'), required=True)
	ap.add_argument("-m", "--mask", nargs=1, help="[3Dimage]", metavar=('*.nii.gz or *.mgh'))
	ap.add_argument("-s", "--scale",action = 'store_true')
	ap.add_argument("--fastica",
		help="Independent component analysis. Input the number of components (e.g.,--fastica 8 for eight components). Outputs the recovered sources, and the component fit for each subject. (recommended to scale first)",
		nargs=1,
		metavar=('INT'))
	return ap

def run(opts):
	if opts.voxel:
		img, img_data = loadnifti(opts.input[0])
	if opts.vertex:
		img, img_data = loadmgh(opts.input[0])
	numMerge=len(opts.input)
	outname=opts.output[0]

	if opts.mask:
		if opts.voxel:
			mask , data_mask = loadnifti(opts.mask[0])
		if opts.vertex:
			mask , data_mask = loadmgh(opts.mask[0])
		mask_index = data_mask>0.99
	else:
		mask_index = np.zeros((img_data.shape[0],img_data.shape[1],img_data.shape[2]))
		mask_index = (mask_index == 0)
	img_data_trunc = img_data[mask_index].astype(np.float32)
	if opts.scale: 
		img_data_trunc = zscaler(img_data_trunc.T).T

	for i in xrange(numMerge):
		print "merging image %s" % opts.input[i]
		if i > 0:
			if opts.voxel:
				_, tempimgdata = loadnifti(opts.input[i])
			if opts.vertex:
				_, tempimgdata = loadmgh(opts.input[i])
			tempimgdata=tempimgdata[mask_index].astype(np.float32)
			if opts.scale:
				tempimgdata = zscaler(tempimgdata.T).T
			img_data_trunc = np.column_stack((img_data_trunc,tempimgdata))
	# remove nan if any
	img_data_trunc[np.isnan(img_data_trunc)] = 0

	if opts.fastica:
		ica = FastICA(n_components=int(opts.fastica[0]),max_iter=5000,  tol=0.0001)
		S_ = ica.fit_transform(img_data_trunc).T
		components = ica.components_.T
		#scaling
		fitcomps = np.copy(S_)
		fitcomps = zscaler(fitcomps)
		img_data_trunc =  np.copy(fitcomps.T) # ram shouldn't be an issue here...
		np.savetxt("%s.ICA_fit.csv" % sys.argv[4],zscaler(components), fmt='%10.8f', delimiter=',')

		#save outputs and ica functions for potential ica removal
		if os.path.exists('ICA_temp'):
			print 'ICA_temp directory exists'
			exit()
		else:
			os.makedirs('ICA_temp')
		np.save('ICA_temp/signals.npy',S_)
		pickle.dump( ica, open( "ICA_temp/icasave.p", "wb" ) )
		if opts.bothhemi:
			lh_tempmask=lh_mask_index*1
			rh_tempmask=rh_mask_index*1
			savemgh(lh_tempmask[lh_tempmask==1], lh_img, lh_mask_index, 'ICA_temp/lh_mask.mgh')
			savemgh(rh_tempmask[rh_tempmask==1], rh_img, rh_mask_index, 'ICA_temp/rh_mask.mgh')
		else:
			tempmask=mask_index*1
			if opts.voxel:
				savenifti(tempmask[tempmask==1], img, mask_index, 'ICA_temp/mask.nii.gz')
			else:
				savemgh(tempmask[tempmask==1], img, mask_index, 'ICA_temp/mask.mgh')


	if opts.voxel:
		savenifti(img_data_trunc.astype(np.float32), img, mask_index, outname)
	if opts.vertex:
		savemgh(img_data_trunc.astype(np.float32), img, mask_index, outname)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)

