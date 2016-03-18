#!/usr/bin/python

import os
import sys
import numpy as np
from time import time
from scipy.stats import linregress
from cython.cy_numstats import calc_beta_se,calc_sobelz
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

if len(sys.argv) < 5:
	print "Usage: %s [start] [stop] [surface (area or thickness)] [mediation type (M, Y, I)]" % (str(sys.argv[0]))
	print "Mediation types: M (image as mediator), Y (image as dependent), I (image as independent)"
else:
	start_time = time()
	np.seterr(divide="ignore", invalid="ignore")

	cmdargs = str(sys.argv)
	arg_perm_start = int(sys.argv[1])
	arg_perm_stop = int(sys.argv[2]) + 1
	surface = str(sys.argv[3])
	medtype = str(sys.argv[4])

#load variables
	ny = np.load("python_temp_med_%s/merge_y.npy" % (surface))
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

#load TFCE fucntion
	calcTFCE_lh = Surf(2, 1, adjac_lh) # H=2, E=1
	calcTFCE_rh = Surf(2, 1, adjac_rh) # H=2, E=1

#permute Sobel Z and write max TFCE values
	if not os.path.exists("output_med_%s/perm_SobelZ_%s" % (surface,medtype)):
		os.mkdir("output_med_%s/perm_SobelZ_%s" % (surface,medtype))
	os.chdir("output_med_%s/perm_SobelZ_%s" % (surface,medtype)) 

	for iter_perm in xrange(arg_perm_start,arg_perm_stop):
		np.random.seed(int(iter_perm*1000+time()))
		print "Iteration number : %d" % (iter_perm)
		indices_perm = np.random.permutation(int(n))
		pathA_nx = pred_x[indices_perm]
		pathB_nx = depend_y[indices_perm]
		if medtype == 'M':
			PathA_beta, PathA_se = calc_beta_se(pathA_nx,ny,n,num_vertex)
			PathB_beta, PathB_se = calc_beta_se(pathB_nx,ny,n,num_vertex)
			SobelZ = calc_sobelz(PathA_beta,PathA_se[1],PathB_beta, PathB_se[1])
		elif medtype == 'Y':
			PathA_beta, _, _, _, PathA_se = linregress(pathA_nx, pathB_nx)
			PathB_beta, PathB_se = calc_beta_se(pathB_nx,ny,n,num_vertex)
			SobelZ = calc_sobelz(PathA_beta,PathA_se,PathB_beta, PathB_se[1])
		elif medtype == 'I':
			PathA_beta, PathA_se = calc_beta_se(pathA_nx,ny,n,num_vertex)
			PathB_beta, _, _, _, PathB_se = linregress(pathA_nx, pathB_nx)
			SobelZ = calc_sobelz(PathA_beta,PathA_se[1],PathB_beta, PathB_se)
		else:
			print "Invalid mediation type"
			exit()

		write_perm_maxTFCE("Zstat_%s" % medtype, SobelZ, num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
		write_perm_maxTFCE("Zstat_%s" % medtype, (SobelZ * -1), num_vertex_lh, bin_mask_lh, bin_mask_rh, all_vertex, calcTFCE_lh, calcTFCE_rh)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))
