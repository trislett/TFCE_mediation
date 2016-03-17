#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
from time import time
from scipy.stats import linregress
from cython.cy_numstats import calc_beta_se,calc_sobelz

start_time = time()
np.seterr(divide="ignore", invalid="ignore")

def write_perm_sobelZ(SobelZ,outdata_mask_lh, affine_mask_lh,outdata_mask_rh, affine_mask_rh, medtype, surface, iter_perm,num_vertex, bin_mask_lh, bin_mask_rh):
	outdata_mask_lh[bin_mask_lh,0,0] = SobelZ[:num_vertex]
	outdata_mask_rh[bin_mask_rh,0,0] = SobelZ[num_vertex:]
	fsurfname_lh = "perm_%d_FS_SobelZ_%s_lh_%s.mgh" % (iter_perm,medtype,surface)
	fslname_lh = "perm_%d_SobelZ_%s_lh_%s" % (iter_perm,medtype,surface)
	fsurfname_rh = "perm_%d_FS_SobelZ_%s_rh_%s.mgh" % (iter_perm,medtype,surface)
	fslname_rh = "perm_%d_SobelZ_%s_rh_%s" % (iter_perm,medtype,surface)
	fslname = "perm_%d_SobelZ_%s_%s" % (iter_perm,medtype,surface)
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask_lh,affine_mask_rh),(fsurfname_lh))
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask_rh,affine_mask_lh),(fsurfname_rh))
	os.system("export FSLOUTPUTTYPE=NIFTI; mri_surf2vol --surfval %s --hemi lh --outvol %s.nii --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat; mri_surf2vol --surfval %s --hemi rh --outvol %s.nii --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat; fslmaths %s -add %s -tfce 2 1 26 %s_TFCE; fslmaths %s -add %s -mul -1 -tfce 2 1 26 neg%s_TFCE; fslstats %s_TFCE -P 100 >> perm_Zstat_TFCE_max_voxel.csv; fslstats neg%s_TFCE -P 100 >> perm_Zstat_TFCE_max_voxel.csv; rm %s %s %s*.nii %s*.nii *%s*.nii" % (fsurfname_lh,fslname_lh,fsurfname_rh,fslname_rh, fslname_lh,fslname_rh,fslname,fslname_lh,fslname_rh,fslname,fslname,fslname, fsurfname_lh, fsurfname_rh, fslname_lh, fslname_rh, fslname))

if len(sys.argv) < 4:
	print "Usage: %s [start] [stop] [surface (area or thickness)] [mediation type (M, Y, I)]" % (str(sys.argv[0]))
	print "Mediation types: M (image as mediator), Y (image as dependent), I (image as independent)"
else:
	cmdargs = str(sys.argv)
	arg_perm_start = int(sys.argv[1])
	arg_perm_stop = int(sys.argv[2]) + 1
	surface = str(sys.argv[3])
	medtype = str(sys.argv[4])

	#merge_y = np.load("python_temp_%s/merge_y.npy" % (surface))
	ny = np.load("python_temp_med_%s/merge_y.npy" % (surface))
	outdata_mask_lh = np.load("python_temp_med_%s/outdata_mask_lh.npy" % (surface))
	affine_mask_lh = np.load("python_temp_med_%s/affine_mask_lh.npy" % (surface))
	outdata_mask_rh = np.load("python_temp_med_%s/outdata_mask_rh.npy" % (surface))
	affine_mask_rh = np.load("python_temp_med_%s/affine_mask_rh.npy" % (surface))
	num_vertex = np.load("python_temp_med_%s/num_vertex.npy" % (surface))
	num_vertex_lh = np.load("python_temp_med_%s/num_vertex_lh.npy" % (surface))
	bin_mask_lh = np.load("python_temp_med_%s/bin_mask_lh.npy" % (surface))
	bin_mask_rh = np.load("python_temp_med_%s/bin_mask_rh.npy" % (surface))
	num_subjects = np.load("python_temp_med_%s/num_subjects.npy" % (surface))
	pred_x = np.load("python_temp_med_%s/pred_x.npy" % (surface))
	depend_y = np.load("python_temp_med_%s/depend_y.npy" % (surface))

	n = len(ny)

	if not os.path.exists("output_med_%s/perm_SobelZ_%s" % (surface,medtype)):
		os.mkdir("output_med_%s/perm_SobelZ_%s" % (surface,medtype))
	os.chdir("output_med_%s/perm_SobelZ_%s" % (surface,medtype)) 

	for iter_perm in xrange(arg_perm_start,arg_perm_stop):
		np.random.seed(int(iter_perm*1000+time()))
		print "Iteration number : %d" % (iter_perm)
		indices_perm = np.random.permutation(n)
		#ny = merge_y[np.random.permutation(range(num_subjects))]
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
		write_perm_sobelZ(SobelZ,outdata_mask_lh, affine_mask_lh,outdata_mask_rh, affine_mask_rh, medtype, surface, iter_perm,num_vertex_lh,bin_mask_lh,bin_mask_rh)
	print("Finished. Randomization took %.1f seconds" % (time() - start_time))
