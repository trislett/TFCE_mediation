#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from cython.cy_numstats import resid_covars,tval_int

def write_tval(tvals, tname, tnum, outdata_mask, affine_mask, surf, hemi, bin_mask):
	outdata_mask[bin_mask,0,0] = tvals
	fsurfname = "FS_tstat_%s_%s_%s.mgh" % (surface,hemi,tname)
	fslname = "tstat_%s_%s_%s" % (surface,hemi,tname)
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)
	os.system = os.popen("mri_surf2vol --surfval %s --hemi %s --outvol %s.nii.gz --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat; fslmaths %s -mul -1 neg%s; fslmaths %s -tfce 2 1 26 %s_TFCE; fslmaths neg%s -tfce 2 1 26 neg%s_TFCE" %(fsurfname,hemi,fslname,fslname,fslname,fslname,fslname,fslname,fslname))

if len(sys.argv) < 4:
	print "Usage: %s [predictor file] [covariate file] [surface (area or thickness)]" % (str(sys.argv[0]))
else:
	cmdargs = str(sys.argv)
	arg_predictor = str(sys.argv[1])
	arg_covars = str(sys.argv[2])
	surface = str(sys.argv[3])
	pred_x = np.genfromtxt(arg_predictor, delimiter=',')
	covars = np.genfromtxt(arg_covars, delimiter=',')
	if not os.path.exists("python_temp_%s" % (surface)):
		os.mkdir("python_temp_%s" % (surface))
	np.save("python_temp_%s/pred_x" % (surface),pred_x)
	np.save("python_temp_%s/covars" % (surface),covars)

	img_data_lh = nib.freesurfer.mghformat.load("lh.all.%s.03B.mgh" % (surface))
	data_full_lh = img_data_lh.get_data()
	data_lh = np.squeeze(data_full_lh)
	affine_mask_lh = img_data_lh.get_affine()
	num_subjects = data_lh.shape[1]
	outdata_mask_lh = np.zeros_like(data_full_lh[:,:,:,1])
	img_mean_lh = nib.freesurfer.mghformat.load("lh.mean.%s.03B.mgh" % (surface))
	mean_full_lh = img_mean_lh.get_data()
	mean_lh = np.squeeze(mean_full_lh)
	bin_mask_lh = mean_lh>0
	data_lh = data_lh[bin_mask_lh]
	num_vertex_lh = data_lh.shape[0]

	img_data_rh = nib.freesurfer.mghformat.load("rh.all.%s.03B.mgh" % (surface))
	data_full_rh = img_data_rh.get_data()
	data_rh = np.squeeze(data_full_rh)
	affine_mask_rh = img_data_rh.get_affine()
	outdata_mask_rh = np.zeros_like(data_full_rh[:,:,:,1])
	img_mean_rh = nib.freesurfer.mghformat.load("rh.mean.%s.03B.mgh" % (surface))
	mean_full_rh = img_mean_rh.get_data()
	mean_rh = np.squeeze(mean_full_rh)
	bin_mask_rh = mean_rh>0
	data_rh = data_rh[bin_mask_rh]
	num_vertex_rh = data_rh.shape[0]

	num_vertex = num_vertex_lh + num_vertex_rh
	np.save("python_temp_%s/num_subjects" % (surface),num_subjects)
	np.save("python_temp_%s/num_vertex" % (surface),num_vertex)
	np.save("python_temp_%s/num_vertex_lh" % (surface),num_vertex_lh)
	np.save("python_temp_%s/num_vertex_rh" % (surface),num_vertex_rh)
	np.save("python_temp_%s/bin_mask_lh" % (surface),bin_mask_lh)
	np.save("python_temp_%s/bin_mask_rh" % (surface),bin_mask_rh)
	np.save("python_temp_%s/affine_mask_lh" % (surface),affine_mask_lh)
	np.save("python_temp_%s/outdata_mask_lh" % (surface),outdata_mask_lh)
	np.save("python_temp_%s/affine_mask_rh" % (surface),affine_mask_rh)
	np.save("python_temp_%s/outdata_mask_rh" % (surface),outdata_mask_rh)

	x_covars = sm.add_constant(covars)
	y_lh = resid_covars(x_covars,data_lh)
	y_rh = resid_covars(x_covars,data_rh)
	merge_y=np.hstack((y_lh,y_rh))
	np.save("python_temp_%s/merge_y" % (surface),merge_y)
	del y_lh
	del y_rh

	n = len(merge_y)
	X = sm.add_constant(pred_x)
	k = len(X.T)
	if not os.path.exists("output_%s" % (surface)):
		os.mkdir("output_%s" % (surface))
	os.chdir("output_%s" % (surface))
	invXX = np.linalg.inv(np.dot(X.T, X))
	tvals=tval_int(X, invXX, merge_y, n, k, num_vertex)

	for j in xrange(k-1):
		tnum=j+1
		write_tval(tvals[tnum,:num_vertex_lh], 'con%d' % tnum, tnum, outdata_mask_lh, affine_mask_lh, surface, 'lh', bin_mask_lh)
		write_tval(tvals[tnum,num_vertex_lh:], 'con%d' % tnum, tnum, outdata_mask_rh, affine_mask_rh, surface, 'rh', bin_mask_rh)
