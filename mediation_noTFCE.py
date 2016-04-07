#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from scipy.stats import linregress
from cython.cy_numstats import resid_covars,calc_beta_se
from joblib import Parallel, delayed
import math
from time import time
from joblib import load, dump

def find_nearest(array,value,p_array,i):
	idx = np.searchsorted(array, value, side="left")
	if idx == len(p_array):
		return p_array[idx-1]
	elif math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
		return p_array[idx-1]
	else:
		return p_array[idx]

def write_sobelZ(SobelZ,outdata_mask, affine, surface, hemi):
	outdata_mask[:,0,0] = SobelZ
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine),"SobelZ_%s_%s_%s.mgh" % (medtype,surface,hemi))

def calc_sobelz(medtype, pred_x, depend_y, merge_y, n, num_vertex):
	if medtype == 'M':
		PathA_beta, PathA_se = calc_beta_se(pred_x,merge_y,n,num_vertex)
		PathB_beta, PathB_se = calc_beta_se(np.column_stack([depend_y,pred_x]),merge_y,n,num_vertex)
		PathA_se = PathA_se[1]
		PathB_se = PathB_se[1]
	elif medtype == 'Y':
		PathA_beta, _, _, _, PathA_se = linregress(pred_x, depend_y)
		PathB_beta, PathB_se = calc_beta_se(np.column_stack([depend_y,pred_x]),merge_y,n,num_vertex)
		PathB_se = PathB_se[1]
	elif medtype == 'I':
		PathA_beta, PathA_se = calc_beta_se(pred_x,merge_y,n,num_vertex)
		PathB_beta, PathB_se = calc_beta_se(np.column_stack([pred_x,depend_y]),merge_y,n,num_vertex)
		PathA_se = PathA_se[1]
		PathB_se = PathB_se[1]
	else:
		print "Invalid mediation type"
		exit()
	ta = PathA_beta/PathA_se
	tb = PathB_beta/PathB_se
	SobelZ = 1/np.sqrt((1/(tb**2))+(1/(ta**2)))
	return SobelZ


def perm_sobelZ(y, pred_x,depend_y, medtype, n, num_vertex, i):
	np.random.seed(int(i*1000+time()))
	print "Iteration number : %d" % (i+1)
	indices_perm = np.random.permutation(n)
	pathA_nx = pred_x[indices_perm]
	pathB_nx = depend_y[indices_perm]
	SobelZ = calc_sobelz(medtype, pathA_nx, pathB_nx, y, n, num_vertex)
	return np.nanmax(SobelZ)

if len(sys.argv) < 6:
	print "Usage: %s [predictor file] [covariate file] [dependent file] [surface (area or thickness)] [mediation type (M, Y, I)] [#perm] [#cores]" % (str(sys.argv[0]))
	print "Mediation types: M (neuoroimage as mediator), Y (neuoroimage as dependent), I (neuroimage as independent)"
else:

	cmdargs = str(sys.argv)
	arg_predictor = str(sys.argv[1])
	arg_covars = str(sys.argv[2])
	arg_depend = str(sys.argv[3])
	surface = str(sys.argv[4])
	medtype = str(sys.argv[5])
	num_perm = int(sys.argv[6])
	num_cores = int(sys.argv[7])
	FWHM = '03B'

	if not os.path.exists("python_temp_med_%s" % surface):
		os.mkdir("python_temp_med_%s" % surface)

	pred_x = np.genfromtxt(arg_predictor, delimiter=",")
	covars = np.genfromtxt(arg_covars, delimiter=",")
	depend_y = np.genfromtxt(arg_depend, delimiter=",")

	np.save("python_temp_med_%s/pred_x" % surface,pred_x)
	np.save("python_temp_med_%s/covars" % surface,covars)
	np.save("python_temp_med_%s/depend_y" % surface,depend_y)

	img_lh = nib.freesurfer.mghformat.load("lh.all.%s.%s.mgh" % (surface,FWHM))
	data_full_lh = img_lh.get_data()
	data_lh = np.squeeze(data_full_lh)
	affine_mask_lh = img_lh.get_affine()
	num_vertex = data_lh.shape[0]
	num_subjects = data_lh.shape[1]
	outdata_mask_lh = data_full_lh[:,:,:,1]

	img_rh = nib.freesurfer.mghformat.load("rh.all.%s.%s.mgh" % (surface,FWHM))
	data_full_rh = img_rh.get_data()
	data_rh = np.squeeze(data_full_rh)
	affine_mask_rh = img_rh.get_affine()
	outdata_mask_rh = data_full_rh[:,:,:,1]

	np.save("python_temp_med_%s/num_subjects" % surface,num_subjects)
	np.save("python_temp_med_%s/num_vertex" % surface,num_vertex)
	np.save("python_temp_med_%s/affine_mask_lh" % surface,affine_mask_lh)
	np.save("python_temp_med_%s/outdata_mask_lh" % surface,outdata_mask_lh)
	np.save("python_temp_med_%s/affine_mask_rh" % surface,affine_mask_rh)
	np.save("python_temp_med_%s/outdata_mask_rh" % surface,outdata_mask_rh)

	x_covars = sm.add_constant(covars)
	y_lh  = resid_covars(x_covars, data_lh)
	y_rh  = resid_covars(x_covars, data_rh)
	np.save("python_temp_med_%s/resids_lh" % surface,y_lh)
	np.save("python_temp_med_%s/resids_rh" % surface,y_rh)
	n = len(y_lh)
	SobelZ_lh = calc_sobelz(medtype, pred_x, depend_y, y_lh, n, num_vertex)
	SobelZ_rh = calc_sobelz(medtype, pred_x, depend_y, y_rh, n, num_vertex)
	SobelZ_lh[np.isnan(SobelZ_lh)] = 0
	SobelZ_rh[np.isnan(SobelZ_rh)] = 0
	write_sobelZ(SobelZ_lh,outdata_mask_lh, affine_mask_lh, surface, 'lh')
	write_sobelZ(SobelZ_rh,outdata_mask_rh, affine_mask_rh, surface, 'rh')
	merge_y = np.hstack((y_lh,y_rh))
	filename = 'joblib_map.mmap'
	if os.path.exists(filename): os.unlink(filename)
	_ = dump(merge_y, filename)
	mapped_merge_y= load(filename, mmap_mode='r')

	del data_lh
	del data_rh
	del data_full_lh
	del data_full_rh
	del y_lh
	del y_rh
	del merge_y
	all_vertex = num_vertex*2
	maxZstat = np.array(Parallel(n_jobs=num_cores)(delayed(perm_sobelZ)(mapped_merge_y,pred_x,depend_y, medtype, n, all_vertex, i) for i in xrange(int(num_perm))))
	maxZstat = np.sort(maxZstat.flatten())
	np.savetxt("maxperm_zvalues.csv",maxZstat, delimiter=",")
	os.system('rm *mmap*')
