#!/usr/bin/env python

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
import numpy as np
import argparse as ap
from time import time

from tfce_mediation.cynumstats import tval_int, calcF
from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.pyfunc import write_perm_maxTFCE_voxel

DESCRIPTION = "Permutation testing for voxel-wise multiple regression with TFCE"
start_time = time()

def getArgumentParser(ap = ap.ArgumentParser(description=DESCRIPTION)):
	ap.add_argument("-r", "--range",
		nargs=2,
		help="permutation [start] [stop]",
		metavar=('INT','INT'),
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

	np.seterr(divide="ignore", invalid="ignore") #only necessary for ANTS skeleton

	#load variables
	num_voxel = np.load('python_temp/num_voxel.npy')
	n = np.load('python_temp/num_subjects.npy')
	ny = np.load('python_temp/raw_nonzero_corr.npy').T
	pred_x = np.load('python_temp/pred_x.npy')
	adjac = np.load('python_temp/adjac.npy')
	ancova = np.load('python_temp/ancova.npy')
	optstfce = np.load('python_temp/optstfce.npy')

	#load TFCE fucntion
	calcTFCE = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac) # H=2, E=2, 26 neighbour connectivity

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
			write_perm_maxTFCE_voxel('fstat', perm_fvals, calcTFCE)
	else:
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
			perm_tvalues=tval_int(nx, invXX, ny, n, k, num_voxel)
			perm_tvalues[np.isnan(perm_tvalues)]=0 #only necessary for ANTS skeleton
			if opts.specifyvars:
				for j in xrange(stop-start):
					tnum=j+1
					write_perm_maxTFCE_voxel('tstat_con%d' % tnum, perm_tvalues[tnum], calcTFCE)
					write_perm_maxTFCE_voxel('tstat_con%d' % tnum, (perm_tvalues[tnum]*-1), calcTFCE)
			else:
				for j in xrange(k-1):
					tnum=j+1
					write_perm_maxTFCE_voxel('tstat_con%d' % tnum, perm_tvalues[tnum], calcTFCE)
					write_perm_maxTFCE_voxel('tstat_con%d' % tnum, (perm_tvalues[tnum]*-1), calcTFCE)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
