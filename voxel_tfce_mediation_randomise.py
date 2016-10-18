#!/usr/bin/python

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
import sys
import numpy as np
import nibabel as nib
from cython.cy_numstats import calc_beta_se
from scipy.stats import linregress
from time import time
from cython.TFCE import Surf
from py_func import write_perm_maxTFCE_voxel, calc_sobelz
import argparse as ap

start_time = time()

ap = ap.ArgumentParser(description="Permutation testing for mediation with TFCE")
ap.add_argument("-r", "--range", nargs=2, help="permutation [start] [stop]", required=True)
ap.add_argument("-m", "--medtype", nargs=1, help="Mediation type [M or Y or I].  Specify which regressors are permuted [first] [last]. For one variable, first=last.", metavar=('INT','INT'))
opts = ap.parse_args()

if len(sys.argv) < 4:
	print "Usage: %s [start] [stop] [mediation type (M, Y, I)]" % (str(sys.argv[0]))
	print "Mediation types: M (image as mediator), Y (image as dependent), I (image as independent)"
else:
	cmdargs = str(sys.argv)
	arg_perm_start = int(sys.argv[1])
	arg_perm_stop = int(sys.argv[2]) + 1
	medtype = str(sys.argv[3])

#load variables
	data_mask = np.load('python_temp/data_mask.npy')
	data_index = data_mask>0.99
	affine_mask = np.load('python_temp/affine_mask.npy')
	num_voxel = np.load('python_temp/num_voxel.npy')
	n = np.load('python_temp/num_subjects.npy')
	ny = np.load('python_temp/raw_nonzero_corr.npy').T
	pred_x = np.load('python_temp/pred_x.npy')
	depend_y = np.load("python_temp/depend_y.npy")
	adjac = np.load('python_temp/adjac.npy')
	optstfce = np.load('python_temp/optstfce.npy')

#load TFCE fucntion
	calcTFCE = Surf(float(optstfce[0]), float(optstfce[1]), adjac) # H=2, E=2, 26 neighbour connectivity

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
