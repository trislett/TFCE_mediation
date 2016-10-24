#!/usr/bin/env python

#    Randomise voxel-based mediation with TFCE
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
import argparse as ap
from time import time

from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.pyfunc import write_perm_maxTFCE_voxel, calc_sobelz

DESCRIPTION = "Permutation testing for voxel-wise mediation with TFCE"
start_time = time()

def getArgumentParser(ap = ap.ArgumentParser(description=DESCRIPTION)):
	ap.add_argument("-r", "--range", 
		nargs=2, 
		help="permutation [start] [stop]", 
		required=True)
	ap.add_argument("-m", "--medtype", 
		nargs=1, 
		help="mediation type [M or Y or I].", 
		choices=['M', 'Y', 'I'], 
		required=True)
	return ap

def run(opts):
	arg_perm_start = int(opts.range[0])
	arg_perm_stop = int(opts.range[1]) + 1
	medtype = str(opts.medtype[0])

	#load variables
	num_voxel = np.load('python_temp/num_voxel.npy')
	n = np.load('python_temp/num_subjects.npy')
	ny = np.load('python_temp/raw_nonzero_corr.npy').T
	pred_x = np.load('python_temp/pred_x.npy')
	depend_y = np.load("python_temp/depend_y.npy")
	adjac = np.load('python_temp/adjac.npy')
	optstfce = np.load('python_temp/optstfce.npy')

	#load TFCE fucntion
	calcTFCE = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac) # H=2, E=2, 26 neighbour connectivity

	#permute Sobel Z values and write max TFCE values
	if not os.path.exists("output_med_%s/perm_SobelZ" % medtype):
		os.mkdir("output_med_%s/perm_SobelZ" % medtype)
	os.chdir("output_med_%s/perm_SobelZ" % medtype)

	for iter_perm in xrange(arg_perm_start,arg_perm_stop):
		np.random.seed(int(iter_perm*1000+time()))
		print "Iteration number : %d" % (iter_perm)
		indices_perm = np.random.permutation(range(n))
		if (medtype == 'M') or (medtype == 'I'):
			pathA_nx = pred_x[indices_perm]
			pathB_nx = depend_y
			SobelZ = calc_sobelz(medtype, pathA_nx, pathB_nx, ny, n, num_voxel)
		else:
			pathA_nx = pred_x[indices_perm]
			pathB_nx = depend_y[indices_perm]
			SobelZ = calc_sobelz(medtype, pathA_nx, pathB_nx, ny, n, num_voxel)
		write_perm_maxTFCE_voxel('Zstat_%s' % medtype, SobelZ, calcTFCE)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)

