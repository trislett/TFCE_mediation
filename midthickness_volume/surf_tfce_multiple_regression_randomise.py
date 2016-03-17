#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
from time import time
from cython.cy_numstats import tval_int
#from joblib import load, dump
import statsmodels.api as sm

# remove divide by zero error
np.seterr(divide="ignore", invalid="ignore")

start_time = time()

def perm_write_tval(tval, tname, tnum, outdata_mask_lh,outdata_mask_rh, affine_mask_lh,affine_mask_rh, surf, iter_perm,num_vertex, bin_mask_lh, bin_mask_rh):
	outdata_mask_lh[bin_mask_lh,0,0] = tvals[tnum,:num_vertex]
	outdata_mask_rh[bin_mask_rh,0,0] = tvals[tnum,num_vertex:]
	fsurfname_lh = "perm_%d_FS_tstat_%s_lh_%s.mgh" % (iter_perm,surface,tname)
	fslname_lh = "perm_%d_tstat_%s_lh_%s" % (iter_perm,surface,tname)
	fsurfname_rh = "perm_%d_FS_tstat_%s_rh_%s.mgh" % (iter_perm,surface,tname)
	fslname_rh = "perm_%d_tstat_%s_rh_%s" % (iter_perm,surface,tname)
	fslname = "perm_%d_tstat_%s_%s" % (iter_perm,surface,tname)
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask_lh,affine_mask_lh),fsurfname_lh)
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask_rh,affine_mask_rh),fsurfname_rh)
	os.system("export FSLOUTPUTTYPE=NIFTI; mri_surf2vol --surfval %s --hemi lh --outvol %s.nii --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat > /dev/null; mri_surf2vol --surfval %s --hemi rh --outvol %s.nii --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat > /dev/null; fslmaths %s -add %s -tfce 2 1 26 %s_TFCE; fslmaths %s -add %s -mul -1 -tfce 2 1 26 neg%s_TFCE; fslstats %s_TFCE -P 100 >> perm_Tstat_TFCE_max_voxel_%s.csv; fslstats neg%s_TFCE -P 100 >> perm_Tstat_TFCE_max_voxel_%s.csv; rm %s %s %s*.nii %s*.nii *%s*.nii" % (fsurfname_lh,fslname_lh,fsurfname_rh,fslname_rh, fslname_lh,fslname_rh,fslname,fslname_lh,fslname_rh,fslname,fslname,tname,fslname,tname,fsurfname_lh, fsurfname_rh, fslname_lh, fslname_rh, fslname))

if len(sys.argv) < 4:
	print "Usage: %s [start] [stop] [surface (area or thickness)]" % (str(sys.argv[0]))
else:
	cmdargs = str(sys.argv)
	arg_perm_start = int(sys.argv[1])
	arg_perm_stop = int(sys.argv[2]) + 1
	surface = str(sys.argv[3])

	ny = np.load("python_temp_%s/merge_y.npy" % (surface))
	outdata_mask_lh = np.load("python_temp_%s/outdata_mask_lh.npy" % (surface))
	affine_mask_lh = np.load("python_temp_%s/affine_mask_lh.npy" % (surface))
	outdata_mask_rh = np.load("python_temp_%s/outdata_mask_rh.npy" % (surface))
	affine_mask_rh = np.load("python_temp_%s/affine_mask_rh.npy" % (surface))
	num_vertex = np.load("python_temp_%s/num_vertex.npy" % (surface))
	num_vertex_lh = np.load("python_temp_%s/num_vertex_lh.npy" % (surface))
	bin_mask_lh = np.load("python_temp_%s/bin_mask_lh.npy" % (surface))
	bin_mask_rh = np.load("python_temp_%s/bin_mask_rh.npy" % (surface))
	n = np.load("python_temp_%s/num_subjects.npy" % (surface))
	pred_x = np.load("python_temp_%s/pred_x.npy" % (surface))

	if not os.path.exists("output_%s/perm_Tstat_%s" % (surface,surface)):
		os.mkdir("output_%s/perm_Tstat_%s" % (surface,surface))
	os.chdir("output_%s/perm_Tstat_%s" % (surface,surface)) 

	X = sm.add_constant(pred_x)
	k = len(X.T)
	for iter_perm in xrange(arg_perm_start,arg_perm_stop):
		np.random.seed(int(iter_perm*1000+time()))
		print "Iteration number : %d" % (iter_perm)
		#ny = merge_y[np.random.permutation(range(n))]
		nx = X[np.random.permutation(range(n))]
		invXX = np.linalg.inv(np.dot(nx.T, nx))
		tvals=tval_int(nx, invXX, ny, n, k, num_vertex)
		for j in xrange(k-1):
			tnum=j+1
			perm_write_tval(tvals,'con%d' % tnum,tnum,outdata_mask_lh,outdata_mask_rh,affine_mask_lh,affine_mask_rh,surface,iter_perm,num_vertex_lh,bin_mask_lh,bin_mask_rh)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))
