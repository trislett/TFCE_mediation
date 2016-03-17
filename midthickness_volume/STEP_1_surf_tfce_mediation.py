#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from scipy.stats import linregress
from cython.cy_numstats import resid_covars,calc_beta_se,calc_sobelz

def write_sobelZ(SobelZ,outdata_mask, affine, surface, hemi, bin_mask):
	outdata_mask[bin_mask,0,0] = SobelZ
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine),"SobelZ_%s_%s.mgh" % (medtype,hemi))
	os.system = os.popen("mri_surf2vol --surfval SobelZ_%s_%s.mgh --hemi %s --outvol SobelZ_%s_%s_%s.nii.gz --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat; fslmaths SobelZ_%s_%s_%s -mul -1 negSobelZ_%s_%s_%s; fslmaths SobelZ_%s_%s_%s -tfce 2 1 26 SobelZ_%s_%s_%s_TFCE; fslmaths negSobelZ_%s_%s_%s -tfce 2 1 26 negSobelZ_%s_%s_%s_TFCE" % (medtype,hemi,hemi,medtype,hemi,surface,medtype,hemi,surface,medtype,hemi,surface,medtype,hemi,surface,medtype,hemi,surface,medtype,hemi,surface,medtype,hemi,surface))

if len(sys.argv) < 6:
	print "Usage: %s [predictor file] [covariate file] [dependent file] [surface (area or thickness)] [mediation type (M, Y, I)]" % (str(sys.argv[0]))
	print "Mediation types: M (neuoroimage as mediator), Y (neuoroimage as dependent), I (neuroimage as independent)"
else:
	cmdargs = str(sys.argv)
	arg_predictor = str(sys.argv[1])
	arg_covars = str(sys.argv[2])
	arg_depend = str(sys.argv[3])
	surface = str(sys.argv[4])
	medtype = str(sys.argv[5])

	if not os.path.exists("python_temp_med_%s" % surface):
		os.mkdir("python_temp_med_%s" % surface)

	pred_x = np.genfromtxt(arg_predictor, delimiter=",")
	covars = np.genfromtxt(arg_covars, delimiter=",")
	depend_y = np.genfromtxt(arg_depend, delimiter=",")

	np.save("python_temp_med_%s/pred_x" % surface,pred_x)
	np.save("python_temp_med_%s/covars" % surface,covars)
	np.save("python_temp_med_%s/depend_y" % surface,depend_y)

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

	np.save("python_temp_med_%s/num_subjects" % surface,num_subjects)
	np.save("python_temp_med_%s/num_vertex" % surface,num_vertex)
	np.save("python_temp_med_%s/num_vertex_lh" % (surface),num_vertex_lh)
	np.save("python_temp_med_%s/num_vertex_rh" % (surface),num_vertex_rh)
	np.save("python_temp_med_%s/bin_mask_lh" % (surface),bin_mask_lh)
	np.save("python_temp_med_%s/bin_mask_rh" % (surface),bin_mask_rh)
	np.save("python_temp_med_%s/affine_mask_lh" % surface,affine_mask_lh)
	np.save("python_temp_med_%s/outdata_mask_lh" % surface,outdata_mask_lh)
	np.save("python_temp_med_%s/affine_mask_rh" % surface,affine_mask_rh)
	np.save("python_temp_med_%s/outdata_mask_rh" % surface,outdata_mask_rh)

	x_covars = sm.add_constant(covars)
	y_lh = resid_covars(x_covars,data_lh)
	y_rh = resid_covars(x_covars,data_rh)
	del data_lh
	del data_rh
	merge_y=np.hstack((y_lh,y_rh))
	np.save("python_temp_med_%s/merge_y" % (surface),merge_y)
	del y_lh
	del y_rh

	n = len(merge_y)

	if not os.path.exists("output_med_%s" % surface):
		os.mkdir("output_med_%s" % surface)
	os.chdir("output_med_%s" % surface)
	
	if medtype == 'M':
		PathA_beta, PathA_se = calc_beta_se(pred_x,merge_y,n,num_vertex)
		PathB_beta, PathB_se = calc_beta_se(depend_y,merge_y,n,num_vertex)
		SobelZ = calc_sobelz(PathA_beta,PathA_se,PathB_beta, PathB_se)
	elif medtype == 'Y':
		PathA_beta, _, _, _, PathA_se = linregress(pred_x, depend_y)
		PathB_beta, PathB_se = calc_beta_se(depend_y,merge_y,n,num_vertex)
		SobelZ = calc_sobelz(PathA_beta,PathA_se,PathB_beta, PathB_se)
	elif medtype == 'I':
		PathA_beta, PathA_se = calc_beta_se(pred_x,merge_y,n,num_vertex)
		PathB_beta, _, _, _, PathB_se = linregress(pred_x, depend_y)
		SobelZ = calc_sobelz(PathA_beta,PathA_se,PathB_beta, PathB_se)
	else:
		print "Invalid mediation type"
		exit()
	write_sobelZ(SobelZ[1,:num_vertex_lh],outdata_mask_lh, affine_mask_lh, surface, 'lh', bin_mask_lh)
	write_sobelZ(SobelZ[1,num_vertex_lh:],outdata_mask_rh, affine_mask_rh, surface, 'rh', bin_mask_rh)
