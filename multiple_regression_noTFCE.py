#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from cython.cy_numstats import resid_covars,tval_int
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

def write_tval(tvals, tname, tnum, outdata_mask, affine_mask, surface, hemi, bin_mask):
	outdata_mask[bin_mask,0,0] = tvals[tnum,:]
	fsurfname = "Tstat_%s_%s_%s.mgh" % (surface,hemi,tname)
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)

def perm_write_tval(tvals, tname, outdata_mask, affine_mask, surface, hemi, stat, bin_mask):
	outdata_mask[bin_mask,0,0] = tvals
	fsurfname = "%s_pFWE_%s_%s_%s.mgh" % (stat,surface,hemi,tname)
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)

def perm_tval(y, X,n,k,num_vertex, i):
	np.random.seed(int(i*1000+time()))
	print "Iteration number : %d" % (i+1)
	nx = X[np.random.permutation(range(n))]
	tvals=tval_int(nx, np.linalg.inv(np.dot(nx.T, nx)), y, n, k, num_vertex)
	return np.hstack((np.nanmax(tvals, axis=1),np.nanmax((tvals*-1), axis=1)))

if len(sys.argv) < 6:
	print "Usage: %s [predictor file] [covariate file] [surface (area or thickness)] [#perm] [#cores]" % (str(sys.argv[0]))
else:

	cmdargs = str(sys.argv)
	arg_predictor = str(sys.argv[1])
	arg_covars = str(sys.argv[2])
	surface = str(sys.argv[3])
	num_perm = int(sys.argv[4])
	num_cores = int(sys.argv[5])

	pred_x = np.genfromtxt(arg_predictor, delimiter=",")
	covars = np.genfromtxt(arg_covars, delimiter=",")


	img_lh = nib.freesurfer.mghformat.load("lh.all.%s.10B.mgh" % surface)
	data_full_lh = img_lh.get_data()
	data_lh = np.squeeze(data_full_lh)
	affine_mask_lh = img_lh.get_affine()
	mean_lh = np.mean(data_lh, axis=1)
	bin_mask_lh = mean_lh>0
	n = data_lh.shape[1]
	outdata_mask_lh = np.zeros_like(data_full_lh[:,:,:,1])
	data_lh = data_lh[bin_mask_lh]
	num_vertex_lh = data_lh.shape[0]

	img_rh = nib.freesurfer.mghformat.load("rh.all.%s.10B.mgh" % surface)
	data_full_rh = img_rh.get_data()
	data_rh = np.squeeze(data_full_rh)
	affine_mask_rh = img_rh.get_affine()
	mean_rh = np.mean(data_rh, axis=1)
	bin_mask_rh = mean_rh>0
	data_rh = data_rh[bin_mask_rh]
	num_vertex_rh = data_rh.shape[0]
	outdata_mask_rh = np.zeros_like(data_full_rh[:,:,:,1])

	num_vertex = num_vertex_lh + num_vertex_rh

	x_covars = sm.add_constant(covars)
	y_lh  = resid_covars(x_covars, data_lh)
	y_rh  = resid_covars(x_covars, data_rh)

	X = sm.add_constant(pred_x)
	k = len(X.T)
	tvals_lh=tval_int(X, np.linalg.inv(np.dot(X.T, X)), y_lh, n, k, num_vertex_lh)
	tvals_rh=tval_int(X, np.linalg.inv(np.dot(X.T, X)), y_rh, n, k, num_vertex_rh)
	tvals_lh[np.isnan(tvals_lh)] = 0
	tvals_rh[np.isnan(tvals_rh)] = 0
	for j in xrange(k-1):
		tnum=j+1
		write_tval(tvals_lh, 'con%d' % tnum, tnum, outdata_mask_lh, affine_mask_lh, surface, 'lh', bin_mask_lh)
		write_tval(tvals_rh, 'con%d' % tnum, tnum, outdata_mask_rh, affine_mask_rh, surface, 'rh', bin_mask_rh)

	maxtfce=np.zeros([(int(num_perm/2)),k*2])
	negmaxtfce=np.zeros([(int(num_perm/2)),k])
	merge_y=np.hstack((y_lh,y_rh))
	filename = 'joblib_map.mmap'
	if os.path.exists(filename): os.unlink(filename)
	_ = dump(merge_y, filename)
	mapped_merge_y = load(filename, mmap_mode='r')
	maxtfce = np.array(Parallel(n_jobs=num_cores)(delayed(perm_tval)(mapped_merge_y,X,n,k,num_vertex, i) for i in xrange(int(num_perm/2))))
	maxtfce = np.vstack((maxtfce[:,:k], maxtfce[:,k:]))
	maxtfce = np.sort(maxtfce,axis=0)
	p_array=np.zeros(maxtfce.shape[0])
	for j in xrange(num_perm):
		p_array[j] = np.true_divide(j,num_perm)
	for j in xrange(k-1):
		tnum=j+1
		tstat_corrp_lh = np.zeros_like(tvals_lh)
		tstat_corrp_rh = np.zeros_like(tvals_rh)
		tstat_corrp_lh = np.array(Parallel(n_jobs=num_cores)(delayed(find_nearest)(maxtfce[:,tnum],tvals_lh[tnum,i],p_array,i) for i in xrange(num_vertex_lh)))
		perm_write_tval(tstat_corrp_lh, 'con%d' % tnum, outdata_mask_lh, affine_mask_lh, surface, 'lh','Tstat', bin_mask_lh)
		tstat_corrp_rh = np.array(Parallel(n_jobs=num_cores)(delayed(find_nearest)(maxtfce[:,tnum],tvals_rh[tnum,i],p_array,i) for i in xrange(num_vertex_rh)))
		perm_write_tval(tstat_corrp_rh, 'con%d' % tnum, outdata_mask_rh, affine_mask_rh, surface, 'rh','Tstat', bin_mask_rh)
		tstat_corrp_lh = np.zeros_like(tvals_lh)
		tstat_corrp_rh = np.zeros_like(tvals_rh)
		tstat_corrp_lh = np.array(Parallel(n_jobs=num_cores)(delayed(find_nearest)(maxtfce[:,tnum],(tvals_lh[tnum,i]*-1),p_array,i) for i in xrange(num_vertex_lh)))
		perm_write_tval(tstat_corrp_lh, 'con%d' % tnum, outdata_mask_lh, affine_mask_lh, surface, 'lh','negTstat', bin_mask_lh)
		tstat_corrp_rh = np.array(Parallel(n_jobs=num_cores)(delayed(find_nearest)(maxtfce[:,tnum],(tvals_rh[tnum,i]*-1),p_array,i) for i in xrange(num_vertex_rh)))
		perm_write_tval(tstat_corrp_rh, 'con%d' % tnum, outdata_mask_rh, affine_mask_rh, surface, 'rh','negTstat', bin_mask_rh)
	np.savetxt("maxperm_tvalues.csv",maxtfce, delimiter=",")
	os.system('rm *mmap*')
