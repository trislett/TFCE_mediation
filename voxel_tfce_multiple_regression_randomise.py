#!/usr/bin/python

#    Randomise voxel-based multiple regression with TFCE
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
from cython.cy_numstats import tval_int, calcF
from time import time
from cython.TFCE import Surf
from py_func import write_perm_maxTFCE

start_time = time()

if len(sys.argv) < 3:
	print "Usage: %s [start] [stop]" % (str(sys.argv[0]))
	print "Note: You must be in same directory as python_temp"
else:
	cmdargs = str(sys.argv)
	arg_perm_start = int(sys.argv[1])
	arg_perm_stop = int(sys.argv[2]) + 1

	np.seterr(divide="ignore", invalid="ignore") #only necessary for ANTS skeleton

#load variables
	data_mask = np.load('python_temp/data_mask.npy')
	data_index = data_mask>0.99
	affine_mask = np.load('python_temp/affine_mask.npy')
	num_voxel = np.load('python_temp/num_voxel.npy')
	n = np.load('python_temp/num_subjects.npy')
	ny = np.load('python_temp/raw_nonzero_corr.npy').T
	pred_x = np.load('python_temp/pred_x.npy')
	adjac = np.load('python_temp/adjac.npy')
	ancova = np.load('python_temp/ancova.npy')
	optstfce = np.load('python_temp/optstfce.npy')

#load TFCE fucntion
	calcTFCE = Surf(float(optstfce[0]), float(optstfce[1]), adjac) # H=2, E=2, 26 neighbour connectivity

#permute T values and write max TFCE values
	if not os.path.exists('output/perm_Tstat'):
		os.mkdir('output/perm_Tstat')
	os.chdir('output/perm_Tstat')

	X = np.column_stack([np.ones(n),pred_x])
	k = len(X.T)
	if ancova ==1:
		for iter_perm in xrange(arg_perm_start, int((arg_perm_stop-1)*2+1)):
			np.random.seed(int(iter_perm*1000+time()))
			print "Permutation number: %d" % (iter_perm)
			nx = X[np.random.permutation(range(n))]
			perm_fvals = calcF(nx, ny, n, k)
			perm_fvals[perm_fvals < 0] = 0
			perm_fvals = np.sqrt(perm_fvals)
			print perm_fvals.max()
			print perm_fvals.min()
			write_perm_maxTFCE('fstat', perm_fvals, calcTFCE)
	else:
		for iter_perm in xrange(arg_perm_start,arg_perm_stop):
			np.random.seed(int(iter_perm*1000+time()))
			print "Iteration number : %d" % (iter_perm)
			nx = X[np.random.permutation(range(n))]
			invXX = np.linalg.inv(np.dot(nx.T, nx))
			perm_tvalues=tval_int(nx, invXX, ny, n, k, num_voxel)
			perm_tvalues[np.isnan(perm_tvalues)]=0 #only necessary for ANTS skeleton
			for j in xrange(k-1):
				tnum=j+1
				write_perm_maxTFCE('tstat_con%d' % tnum, perm_tvalues[tnum], calcTFCE)
				write_perm_maxTFCE('tstat_con%d' % tnum, (perm_tvalues[tnum]*-1), calcTFCE)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))