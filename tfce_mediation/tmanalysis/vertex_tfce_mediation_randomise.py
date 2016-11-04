#!/usr/bin/env python

#    Randomise vertex-based mediation with TFCE
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

import os
import numpy as np
from time import time
import argparse as ap

from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.pyfunc import write_perm_maxTFCE_vertex, calc_sobelz

DESCRIPTION = "Permutation testing for vetex-wise mediation with TFCE"
start_time = time()
np.seterr(divide="ignore", invalid="ignore")

def getArgumentParser(ap = ap.ArgumentParser(description=DESCRIPTION)):
	ap.add_argument("-r", "--range", 
		nargs=2, 
		help="permutation [start] [stop]", 
		metavar=('INT', 'INT'), 
		required=True)
	ap.add_argument("-s", "--surface", 
		nargs=1, 
		help="surface (area or thickness)", 
		metavar=('STR'), 
		required=True)
	ap.add_argument("-m", "--medtype", 
		nargs=1, help="mediation type [M or Y or I].", 
		choices=['M', 'Y', 'I'], 
		required=True)
	return ap

def run(opts):
	arg_perm_start = int(opts.range[0])
	arg_perm_stop = int(opts.range[1]) + 1
	medtype = str(opts.medtype[0])
	surface = str(opts.surface[0])

	#load variables
	y = np.load("python_temp_med_%s/merge_y.npy" % (surface))
	num_vertex = np.load("python_temp_med_%s/num_vertex.npy" % (surface))
	num_vertex_lh = np.load("python_temp_med_%s/num_vertex_lh.npy" % (surface))
	bin_mask_lh = np.load("python_temp_med_%s/bin_mask_lh.npy" % (surface))
	bin_mask_rh = np.load("python_temp_med_%s/bin_mask_rh.npy" % (surface))
	n = np.load("python_temp_med_%s/num_subjects.npy" % (surface))
	pred_x = np.load("python_temp_med_%s/pred_x.npy" % (surface))
	depend_y = np.load("python_temp_med_%s/depend_y.npy" % (surface))
	adjac_lh = np.load("python_temp_med_%s/adjac_lh.npy" % (surface))
	adjac_rh = np.load("python_temp_med_%s/adjac_rh.npy" % (surface))
	all_vertex = np.load("python_temp_med_%s/all_vertex.npy" % (surface))
	optstfce = np.load('python_temp_med_%s/optstfce.npy' % (surface))

	#load TFCE fucntion
	calcTFCE_lh = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac_lh) # H=2, E=1
	calcTFCE_rh = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac_rh) # H=2, E=1

	#permute Sobel Z
	if not os.path.exists("output_med_%s/perm_SobelZ_%s" % (surface,medtype)):
		os.mkdir("output_med_%s/perm_SobelZ_%s" % (surface,medtype))
	os.chdir("output_med_%s/perm_SobelZ_%s" % (surface,medtype)) 

	for iter_perm in xrange(arg_perm_start,arg_perm_stop):
		np.random.seed(int(iter_perm*1000+time()))
		print "Iteration number : %d" % (iter_perm)
		indices_perm = np.random.permutation(range(n))
		if (medtype == 'M') or (medtype == 'I'):
			pathA_nx = pred_x[indices_perm]
			pathB_nx = depend_y
			SobelZ = calc_sobelz(medtype, pathA_nx, pathB_nx, y, n, num_vertex)
		else:
			pathA_nx = pred_x[indices_perm]
			pathB_nx = depend_y[indices_perm]
			SobelZ = calc_sobelz(medtype, pathA_nx, pathB_nx, y, n, num_vertex)
		write_perm_maxTFCE_vertex("Zstat_%s" % medtype, SobelZ, num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
