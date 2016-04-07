#!/usr/bin/python

#    Randomise vertex-based multiple regression with TFCE
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
from time import time
from cython.cy_numstats import tval_int
from cython.TFCE import Surf
from py_func import write_perm_maxTFCE

if len(sys.argv) < 4:
	print "Usage: %s [start] [stop] [surface (area or thickness)]" % (str(sys.argv[0]))
else:
	start_time = time()
	np.seterr(divide="ignore", invalid="ignore")

	cmdargs = str(sys.argv)
	arg_perm_start = int(sys.argv[1])
	arg_perm_stop = int(sys.argv[2]) + 1
	surface = str(sys.argv[3])

#load variables
	ny = np.load("python_temp_%s/merge_y.npy" % (surface))
	num_vertex = np.load("python_temp_%s/num_vertex.npy" % (surface))
	num_vertex_lh = np.load("python_temp_%s/num_vertex_lh.npy" % (surface))
	all_vertex = np.load("python_temp_%s/all_vertex.npy" % (surface))
	bin_mask_lh = np.load("python_temp_%s/bin_mask_lh.npy" % (surface))
	bin_mask_rh = np.load("python_temp_%s/bin_mask_rh.npy" % (surface))
	n = np.load("python_temp_%s/num_subjects.npy" % (surface))
	pred_x = np.load("python_temp_%s/pred_x.npy" % (surface))
	adjac_lh = np.load("python_temp_%s/adjac_lh.npy" % (surface))
	adjac_rh = np.load("python_temp_%s/adjac_rh.npy" % (surface))

#load TFCE fucntion
	calcTFCE_lh = Surf(2, 1, adjac_lh) # H=2, E=1
	calcTFCE_rh = Surf(2, 1, adjac_rh) # H=2, E=1

#permute T values and write max TFCE values
	if not os.path.exists("output_%s/perm_Tstat_%s" % (surface,surface)):
		os.mkdir("output_%s/perm_Tstat_%s" % (surface,surface))
	os.chdir("output_%s/perm_Tstat_%s" % (surface,surface)) 

	X = np.column_stack([np.ones(n),pred_x])
	k = len(X.T)
	for iter_perm in xrange(arg_perm_start,arg_perm_stop):
		np.random.seed(int(iter_perm*1000+time()))
		print "Iteration number : %d" % (iter_perm)
		nx = X[np.random.permutation(range(n))]
		invXX = np.linalg.inv(np.dot(nx.T, nx))
		tvals=tval_int(nx, invXX, ny, n, k, num_vertex)
		for j in xrange(k-1):
			tnum=j+1
			write_perm_maxTFCE('tstat_con%d' % tnum, tvals[tnum], num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
			write_perm_maxTFCE('tstat_con%d' % tnum, (tvals[tnum] * -1), num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))
