#!/usr/bin/env python

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
import numpy as np
from time import time
import argparse as ap

from tfce_mediation.cynumstats import tval_int
from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.pyfunc import write_perm_maxTFCE_vertex

DESCRIPTION = "Permutation testing for vertex-wise multiple regression with TFCE"
start_time = time()
np.seterr(divide="ignore", invalid="ignore")

def getArgumentParser(ap = ap.ArgumentParser(description=DESCRIPTION)):
	ap.add_argument("-r", "--range", 
		nargs=2, 
		type=int, 
		help="permutation [start] [stop]", 
		metavar=('INT','INT'), 
		required=True)
	ap.add_argument("-s", "--surface", 
		nargs=1, 
		help="surface (area or thickness)", 
		metavar=('STR'), 
		required=True)
	ap.add_argument("-v", "--specifyvars", 
		nargs=2, 
		type=int, 
		help="Optional. Specify which regressors are permuted [first] [last]. For one variable, first=last.", 
		metavar=('INT','INT'))
	return ap

def run(opts):
	arg_perm_start = int(opts.range[0])
	arg_perm_stop = int(opts.range[1]) + 1
	surface = str(opts.surface[0])

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
	optstfce = np.load('python_temp_%s/optstfce.npy' % (surface))

	#load TFCE fucntion
	calcTFCE_lh = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac_lh) # H=2, E=1
	calcTFCE_rh = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac_rh) # H=2, E=1

	#permute T values and write max TFCE values
	if not os.path.exists("output_%s/perm_Tstat_%s" % (surface,surface)):
		os.mkdir("output_%s/perm_Tstat_%s" % (surface,surface))
	os.chdir("output_%s/perm_Tstat_%s" % (surface,surface)) 

	X = np.column_stack([np.ones(n),pred_x])
	k = len(X.T)
	for iter_perm in xrange(arg_perm_start,arg_perm_stop):
		np.random.seed(int(iter_perm*1000+time()))
		print "Iteration number : %d" % (iter_perm)
		if opts.specifyvars:
			start=opts.specifyvars[0]
			stop=opts.specifyvars[1]+1
			nx = X
			nx[:,start:stop]=X[:,start:stop][np.random.permutation(range(n))]
		else:
			nx = X[np.random.permutation(range(n))]
		invXX = np.linalg.inv(np.dot(nx.T, nx))
		tvals=tval_int(nx, invXX, ny, n, k, num_vertex)
		if opts.specifyvars:
			for j in xrange(stop-start):
				tnum=j+1
				write_perm_maxTFCE_vertex('tstat_con%d' % tnum, tvals[tnum], num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
				write_perm_maxTFCE_vertex('tstat_con%d' % tnum, (tvals[tnum] * -1), num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
		else:
			for j in xrange(k-1):
				tnum=j+1
				write_perm_maxTFCE_vertex('tstat_con%d' % tnum, tvals[tnum], num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
				write_perm_maxTFCE_vertex('tstat_con%d' % tnum, (tvals[tnum] * -1), num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
