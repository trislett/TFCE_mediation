#!/usr/bin/python

#    Calculate 1-PFWE images of vertex-based *.mgh from max TFCE file
#    Copyright (C) 2016  Tristram Lett

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

import numpy as np
import nibabel as nib
import argparse as ap
import math

DESCRIPTION = "Calculate 1-P[FWE] surface from max TFCE values from randomisation."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i", "--input", 
		nargs=2,
		metavar=('*.mgh', '*.csv'),
		help="[tfce_image] [perm_tfce_max]",
		required=True)
	return ap

#find nearest permuted TFCE max value that corresponse to family-wise error rate 
def find_nearest(array,value,p_array):
	idx = np.searchsorted(array, value, side="left")
	if idx == len(p_array):
		return p_array[idx-1]
	elif math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
		return p_array[idx-1]
	else:
		return p_array[idx]


def run(opts):
	arg_tfce_image = str(opts.input[0])
	arg_maxTFCE = str(opts.input[1])
	perm_tfce_max = np.genfromtxt(arg_maxTFCE, delimiter=',')

#load data
	img = nib.freesurfer.mghformat.load(arg_tfce_image)
	arg_tfce_image_noext = arg_tfce_image.split('.mgh',1)[0]
	data_full = img.get_data()
	data = np.squeeze(data_full)
	affine_mask = img.get_affine()
	bin_mask = data>0
	masked_data=data[bin_mask]
	num_perm=perm_tfce_max.shape[0]

#sort max tfce values
	sorted_perm_tfce_max=np.sort(perm_tfce_max)
	p_array=np.zeros_like(sorted_perm_tfce_max)
	corrp_img = np.zeros(masked_data.shape)
	for j in xrange(num_perm):
		p_array[j] = np.true_divide(j,num_perm)
	cV=0
	for k in masked_data:
		corrp_img[cV] = find_nearest(sorted_perm_tfce_max,k,p_array)
		cV+=1

#output corrected image,and printout accuracy based on number of permuations
	outmask=np.zeros_like(data_full)
	outmask[bin_mask,0,0]=corrp_img
	nib.save(nib.Nifti1Image(outmask,affine_mask),"%s_FWEcorrP.mgh" % (arg_tfce_image_noext))
	print "The accuracy is p = 0.05 +/- %.4f" % (2*(np.sqrt(0.05*0.95/num_perm)))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
