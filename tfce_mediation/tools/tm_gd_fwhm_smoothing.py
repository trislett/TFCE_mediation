#!/usr/bin/env python

#    tm_gd_fwhm_smoothing: geodesic FHWM smoothing using accurate distances at midthickness surface
#    Copyright (C) 2016 Tristram Lett, Lea Waller

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

from __future__ import division
import os
import numpy as np
import nibabel as nib
import argparse as ap
from scipy.stats import norm

from tfce_mediation.pyfunc import loadmgh
from tfce_mediation.cynumstats import calc_gd_fwhm

DESCRIPTION = "Geodesic FHWM smoothing using accurate distances at midthickness surface. Expect processing to take ~4 minutes."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	ap.add_argument("-i", "--input", 
		nargs=1, 
		help="input surface image to smooth", 
		metavar=('*.mgh'), 
		required=True)
	ap.add_argument("-o", "--output", 
		nargs=1, 
		help="[4D_image]", 
		metavar=('*.mgh'), 
		required=True)
	ap.add_argument("--hemi", 
		nargs=1, 
		help="hemisphere (e.g., --hemi ?h)", 
		metavar=('lh or rh'), 
		required=True)
	ap.add_argument("-f", "--geodesicfwhm", 
		help="Geodesic FWHM value", 
		type=float,
		nargs=1, 
		default=[3.0],
		metavar=('FLOAT'))
	ap.add_argument("-d","--distanceslist", 
		nargs=1, 
		help="Input the precomputed fwhm distances (can be downloaded from tm_addons). Note, the *_indices.npy file must be in the same directory.", 
		metavar=('/path/to/?h_8.0mm_fwhm_distances.npy'), 
		required=True)
	ap.add_argument("--correct_surface", 
		nargs=1,
		help="Experimental. Corrects extreme outliers. Enter the number of standard deviations (sigmas) constitutes an outlier (e.g. for 3 standard deviatiosn, --correct_surface 3.0). This should not be used with after performing a Box-Cox transformation.", 
		type=float, 
		metavar=('FLOAT'))
	return ap

def run(opts):

	scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	img, data = loadmgh(opts.input[0])
	outname = str(opts.output[0])
	hemi = str(opts.hemi[0])
	fwhm = float(opts.geodesicfwhm[0])
	sigma = fwhm / np.sqrt(8 * np.log(2))

	# load distances and indices 
	dist = np.load(opts.distanceslist[0])
	basenamefwhm = opts.distanceslist[0].split('_distances.npy',1)[0]
	indices = np.load('%s_indices.npy' % basenamefwhm)

	cortex_index = np.load('%s/adjacency_sets/%s_cortex_mask_index.npy' % (scriptwd,hemi))
	distmask_index_start = np.in1d(indices[:,0], cortex_index)
	distmask_index_end = np.in1d(indices[:,1], cortex_index)
	distmask_index = distmask_index_start & distmask_index_end

	# mask out non-cortex values
	indices_masked = indices[distmask_index]
	dist_masked = dist[distmask_index]

	if opts.correct_surface:
		multipler = opts.correct_surface[0]
		y = data[cortex_index].flatten()
		(mu, sigma) = norm.fit(y)
		cthresh = multipler*sigma + mu
		print "The upper threshold is: %1.4f" % cthresh
		data[data>cthresh] = cthresh

	smoothed = calc_gd_fwhm(indices_masked.astype(np.int32, order='c'), dist_masked.astype(np.float32, order='c'), data[:,0,0].astype(np.float32, order = "c"), sigma)

	outdata = np.zeros_like(data)
	outdata[cortex_index,0,0] = smoothed[cortex_index]
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata,img.affine),outname)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)

