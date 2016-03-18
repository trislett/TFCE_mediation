#!/usr/bin/python

import os
import sys
import numpy as np
from time import time
from cython.cy_numstats import tval_int
from cython.TFCE import Surf

def write_perm_maxTFCE(statname, vertStat, num_vertex, bin_mask_lh, bin_mask_rh, all_vertex,calcTFCE_lh,calcTFCE_rh):
	vertStat_out_lh=np.zeros(all_vertex).astype(np.float32, order = "C")
	vertStat_out_rh=np.zeros(all_vertex).astype(np.float32, order = "C")
	vertStat_TFCE_lh = np.zeros_like(vertStat_out_lh).astype(np.float32, order = "C")
	vertStat_TFCE_rh = np.zeros_like(vertStat_out_rh).astype(np.float32, order = "C")
	vertStat_out_lh[bin_mask_lh] = vertStat[:num_vertex]
	vertStat_out_rh[bin_mask_rh] = vertStat[num_vertex:]
	calcTFCE_lh.run(vertStat_out_lh, vertStat_TFCE_lh)
	calcTFCE_rh.run(vertStat_out_rh, vertStat_TFCE_rh)
	maxTFCE = np.array([(vertStat_TFCE_lh.max()*(vertStat_out_lh.max()/100)),(vertStat_TFCE_rh.max()*(vertStat_out_rh.max()/100))]).max() 
	os.system("echo %.4f >> perm_%s_TFCE_maxVoxel.csv" % (maxTFCE,statname))

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
