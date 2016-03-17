#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from scipy.stats import linregress
from cython.cy_numstats import resid_covars,calc_beta_se,calc_sobelz
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
	outdata_mask[:,0,0] = SobelZ[1]
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine),"SobelZ_%s_%s_%s.mgh" % (medtype,surface,hemi))

def write_sobelCorrp(sobelZ_corrp,outdata_mask, affine, surface, hemi, zname):
	outdata_mask[:,0,0] = sobelZ_corrp
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine),"%s_pFWE_%s_%s_%s.mgh" % (zname,medtype,surface,hemi))

def perm_sobelZ(y, pred_x,depend_y, medtype, i):
	np.random.seed(int(i*1000+time()))
	print "Iteration number : %d" % (i+1)
	indices_perm = np.random.permutation(n)
	pathA_nx = pred_x[indices_perm]
	pathB_nx = depend_y[indices_perm]
	if medtype == 'M':
		PathA_beta, PathA_se = calc_beta_se(pathA_nx,y,n,num_vertex)
		PathB_beta, PathB_se = calc_beta_se(pathB_nx,y,n,num_vertex)
		SobelZ = calc_sobelz(PathA_beta, PathA_se,PathB_beta, PathB_se)
	elif medtype == 'Y':
		PathA_beta, intercept, r_value, p_value, PathA_se = linregress(pathA_nx, pathB_nx)
		PathB_beta, PathB_se = calc_beta_se(pathB_nx,y,n,num_vertex)
		SobelZ = calc_sobelz(PathA_beta, PathA_se,PathB_beta, PathB_se)
	elif medtype == 'I':
		PathA_beta, PathA_se = calc_beta_se(pathA_nx,y,n,num_vertex)
		PathB_beta, intercept, r_value, p_value, PathB_se = linregress(pathA_nx, pathB_nx)
		SobelZ = calc_sobelz(PathA_beta, PathA_se,PathB_beta, PathB_se)
	else:
		print "Invalid mediation type"
		exit()
	return np.nanmax(SobelZ[1]),np.nanmax(-1*SobelZ[1])

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

	if not os.path.exists("python_temp_med_%s" % surface):
		os.mkdir("python_temp_med_%s" % surface)

	pred_x = np.genfromtxt(arg_predictor, delimiter=",")
	covars = np.genfromtxt(arg_covars, delimiter=",")
	depend_y = np.genfromtxt(arg_depend, delimiter=",")

	np.save("python_temp_med_%s/pred_x" % surface,pred_x)
	np.save("python_temp_med_%s/covars" % surface,covars)
	np.save("python_temp_med_%s/depend_y" % surface,depend_y)

	img_lh = nib.freesurfer.mghformat.load("lh.all.%s.10B.mgh" % surface)
	data_full_lh = img_lh.get_data()
	data_lh = np.squeeze(data_full_lh)
	affine_mask_lh = img_lh.get_affine()
	num_vertex = data_lh.shape[0]
	num_subjects = data_lh.shape[1]
	outdata_mask_lh = data_full_lh[:,:,:,1]

	img_rh = nib.freesurfer.mghformat.load("rh.all.%s.10B.mgh" % surface)
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
	if medtype == 'M':
		PathA_beta_lh, PathA_se_lh = calc_beta_se(pred_x,y_lh,n,num_vertex)
		PathA_beta_rh, PathA_se_rh = calc_beta_se(pred_x,y_rh,n,num_vertex)
		PathB_beta_lh, PathB_se_lh = calc_beta_se(depend_y,y_lh,n,num_vertex)
		PathB_beta_rh, PathB_se_rh = calc_beta_se(depend_y,y_rh,n,num_vertex)
		SobelZ_lh = calc_sobelz(PathA_beta_lh,PathA_se_lh,PathB_beta_lh, PathB_se_lh)
		SobelZ_rh = calc_sobelz(PathA_beta_rh,PathA_se_rh,PathB_beta_rh, PathB_se_rh)
	elif medtype == 'Y':
		PathA_beta, intercept, r_value, p_value, PathA_se = linregress(pred_x, depend_y)
		PathB_beta_lh, PathB_se_lh = calc_beta_se(depend_y,y_lh,n,num_vertex)
		PathB_beta_rh, PathB_se_rh = calc_beta_se(depend_y,y_rh,n,num_vertex)
		SobelZ_lh = calc_sobelz(PathA_beta,PathA_se,PathB_beta_lh, PathB_se_lh)
		SobelZ_rh = calc_sobelz(PathA_beta,PathA_se,PathB_beta_rh, PathB_se_rh)
	elif medtype == 'I':
		PathA_beta_lh, PathA_se_lh = calc_beta_se(pred_x,y_lh,n,num_vertex)
		PathA_beta_rh, PathA_se_rh = calc_beta_se(pred_x,y_rh,n,num_vertex)
		PathB_beta, intercept, r_value, p_value, PathB_se = linregress(pred_x, depend_y)
		SobelZ_lh = calc_sobelz(PathA_beta_lh,PathA_se_lh,PathB_beta, PathB_se)
		SobelZ_rh = calc_sobelz(PathA_beta_rh,PathA_se_rh,PathB_beta, PathB_se)
	else:
		print "Invalid mediation type"
		exit()
	SobelZ_lh[np.isnan(SobelZ_lh)] = 0
	SobelZ_rh[np.isnan(SobelZ_rh)] = 0
	write_sobelZ(SobelZ_lh,outdata_mask_lh, affine_mask_lh, surface, 'lh')
	write_sobelZ(SobelZ_rh,outdata_mask_rh, affine_mask_rh, surface, 'rh')
	maxZstat_lh=np.zeros([(int(num_perm/2)),2])
	maxZstat_rh=np.zeros([(int(num_perm/2)),2])
	merge_y=np.hstack((y_lh,y_rh))
	filename = 'joblib_map.mmap'
	if os.path.exists(filename): os.unlink(filename)
	_ = dump(merge_y, filename)
	mapped_merge_y= load(filename, mmap_mode='r')
	maxZstat = np.array(Parallel(n_jobs=num_cores)(delayed(perm_sobelZ)(mapped_merge_y,pred_x,depend_y, medtype, i) for i in xrange(int(num_perm/2))))
	maxZstat = np.sort(maxZstat.flatten())
	np.savetxt("maxperm_zvalues.csv",maxZstat, delimiter=",")
	sobelZ_corrp_lh = np.zeros(num_vertex)
	sobelZ_corrp_rh = np.zeros(num_vertex)
	p_array=np.zeros(maxZstat.shape)
	for j in xrange(num_perm):
		p_array[j] = np.true_divide(j,num_perm)
	sobelZ_corrp_lh = np.array(Parallel(n_jobs=num_cores)(delayed(find_nearest)(maxZstat,SobelZ_lh[1,i],p_array,i) for i in xrange(num_vertex)))
	sobelZ_corrp_rh = np.array(Parallel(n_jobs=num_cores)(delayed(find_nearest)(maxZstat,SobelZ_rh[1,i],p_array,i) for i in xrange(num_vertex)))
	write_sobelCorrp(sobelZ_corrp_lh,outdata_mask_lh, affine_mask_lh, surface, 'lh', 'SobelZ')
	write_sobelCorrp(sobelZ_corrp_rh,outdata_mask_rh, affine_mask_rh, surface, 'rh', 'SobelZ')
	sobelZ_corrp_lh = np.array(Parallel(n_jobs=num_cores)(delayed(find_nearest)(maxZstat,(SobelZ_lh[1,i]*-1),p_array,i) for i in xrange(num_vertex)))
	sobelZ_corrp_rh = np.array(Parallel(n_jobs=num_cores)(delayed(find_nearest)(maxZstat,(SobelZ_rh[1,i]*-1),p_array,i) for i in xrange(num_vertex)))
	write_sobelCorrp(sobelZ_corrp_lh,outdata_mask_lh, affine_mask_lh, surface, 'lh', 'negSobelZ')
	write_sobelCorrp(sobelZ_corrp_rh,outdata_mask_rh, affine_mask_rh, surface, 'rh', 'negSobelZ')
	os.system('rm *mmap*')
