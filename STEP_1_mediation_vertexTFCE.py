#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
from scipy.stats import linregress
from cython.cy_numstats import resid_covars,calc_beta_se,calc_sobelz
from cython.TFCE import Surf

def write_vertStat_img(statname, vertStat, outdata_mask, affine_mask, surf, hemi, bin_mask, TFCEfunc, all_vertex):
	vertStat_out=np.zeros(all_vertex).astype(np.float32, order = "C")
	vertStat_out[bin_mask] = vertStat
	vertStat_TFCE = np.zeros_like(vertStat_out).astype(np.float32, order = "C")
	TFCEfunc.run(vertStat_out, vertStat_TFCE)
	outdata_mask[:,0,0] = vertStat_TFCE * (vertStat.max()/100)
	fsurfname = "%s_%s_%s_TFCE.mgh" % (statname,surf,hemi)
	os.system("echo %s_%s_%s,%f >> max_contrast_value.csv" % (statname,surf,hemi, outdata_mask[:,0,0].max()))
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)

def create_adjac (vertices,faces):
	adjacency = [set([]) for i in xrange(vertices.shape[0])]
	for i in xrange(faces.shape[0]):
		adjacency[faces[i, 0]].add(faces[i, 1])
		adjacency[faces[i, 0]].add(faces[i, 2])
		adjacency[faces[i, 1]].add(faces[i, 0])
		adjacency[faces[i, 1]].add(faces[i, 2])
		adjacency[faces[i, 2]].add(faces[i, 0])
		adjacency[faces[i, 2]].add(faces[i, 1])
	return adjacency

if len(sys.argv) < 6:
	print "Usage: %s [predictor file] [covariate file] [dependent file] [surface (area or thickness)] [mediation type (M, Y, I)] optional: [lh_adjacency_dist_?mm.npy] [rh_adjacency_dist_?mm.npy]" % (str(sys.argv[0]))
	print "Mediation types: M (neuroimage as mediator), Y (neuroimage as dependent), I (neuroimage as independent)"
else:
	cmdargs = str(sys.argv)
	arg_predictor = str(sys.argv[1])
	arg_covars = str(sys.argv[2])
	arg_depend = str(sys.argv[3])
	surface = str(sys.argv[4])
	medtype = str(sys.argv[5])

#load variables
	pred_x = np.genfromtxt(arg_predictor, delimiter=",")
	covars = np.genfromtxt(arg_covars, delimiter=",")
	depend_y = np.genfromtxt(arg_depend, delimiter=",")

#load data
	img_data_lh = nib.freesurfer.mghformat.load("lh.all.%s.03B.mgh" % (surface))
	data_full_lh = img_data_lh.get_data()
	data_lh = np.squeeze(data_full_lh)
	affine_mask_lh = img_data_lh.get_affine()
	num_subjects = data_lh.shape[1]
	outdata_mask_lh = np.zeros_like(data_full_lh[:,:,:,1])
	img_data_rh = nib.freesurfer.mghformat.load("rh.all.%s.03B.mgh" % (surface))
	data_full_rh = img_data_rh.get_data()
	data_rh = np.squeeze(data_full_rh)
	affine_mask_rh = img_data_rh.get_affine()
	outdata_mask_rh = np.zeros_like(data_full_rh[:,:,:,1])
	if not os.path.exists("lh.mean.%s.03B.mgh" % (surface)):
		mean_lh = np.sum(data_lh,axis=1)/data_lh.shape[1]
		outmean_lh = np.zeros_like(data_full_lh[:,:,:,1])
		outmean_lh[:,0,0] = mean_lh
		nib.save(nib.freesurfer.mghformat.MGHImage(outmean_lh,affine_mask_lh),"lh.mean.%s.03B.mgh" % (surface))
		mean_rh = np.sum(data_rh,axis=1)/data_rh.shape[1]
		outmean_rh = np.zeros_like(data_full_rh[:,:,:,1])
		outmean_rh[:,0,0] = mean_rh
		nib.save(nib.freesurfer.mghformat.MGHImage(outmean_rh,affine_mask_rh),"rh.mean.%s.03B.mgh" % (surface))
	else:
		img_mean_lh = nib.freesurfer.mghformat.load("lh.mean.%s.03B.mgh" % (surface))
		mean_full_lh = img_mean_lh.get_data()
		mean_lh = np.squeeze(mean_full_lh)
		img_mean_rh = nib.freesurfer.mghformat.load("rh.mean.%s.03B.mgh" % (surface))
		mean_full_rh = img_mean_rh.get_data()
		mean_rh = np.squeeze(mean_full_rh)

#create masks
	bin_mask_lh = mean_lh>0
	data_lh = data_lh[bin_mask_lh]
	num_vertex_lh = data_lh.shape[0]
	bin_mask_rh = mean_rh>0
	data_rh = data_rh[bin_mask_rh]
	num_vertex_rh = data_rh.shape[0]
	num_vertex = num_vertex_lh + num_vertex_rh
	all_vertex = data_full_lh.shape[0]

#TFCE
	if len(sys.argv) == 6:
		print "Creating adjacency set"
		# 3 Neighbour vertex connectity
		v_lh, faces_lh = nib.freesurfer.read_geometry("%s/fsaverage/surf/lh.sphere" % os.environ["SUBJECTS_DIR"])
		v_rh, faces_rh = nib.freesurfer.read_geometry("%s/fsaverage/surf/rh.sphere" % os.environ["SUBJECTS_DIR"])
		adjac_lh = create_adjac(v_lh,faces_lh)
		adjac_rh = create_adjac(v_rh,faces_rh)
	elif len(sys.argv) == 8:
		print "Loading prior adjacency set"
		arg_adjac_lh = str(sys.argv[4])
		arg_adjac_rh = str(sys.argv[5])
		adjac_lh = np.load(arg_adjac_lh)
		adjac_rh = np.load(arg_adjac_rh)
	else:
		print "Error loading adjacency sets"
	calcTFCE_lh = Surf(2, 1, adjac_lh) # H=2, E=1
	calcTFCE_rh = Surf(2, 1, adjac_rh) # H=2, E=1

#save variables
	if not os.path.exists("python_temp_med_%s" % surface):
		os.mkdir("python_temp_med_%s" % surface)

	np.save("python_temp_med_%s/pred_x" % surface,pred_x)
	np.save("python_temp_med_%s/covars" % surface,covars)
	np.save("python_temp_med_%s/depend_y" % surface,depend_y)
	np.save("python_temp_med_%s/num_subjects" % surface,num_subjects)
	np.save("python_temp_med_%s/num_vertex" % surface,num_vertex)
	np.save("python_temp_med_%s/num_vertex_lh" % (surface),num_vertex_lh)
	np.save("python_temp_med_%s/num_vertex_rh" % (surface),num_vertex_rh)
	np.save("python_temp_med_%s/all_vertex" % (surface),all_vertex)
	np.save("python_temp_med_%s/bin_mask_lh" % (surface),bin_mask_lh)
	np.save("python_temp_med_%s/bin_mask_rh" % (surface),bin_mask_rh)
	np.save("python_temp_med_%s/adjac_lh" % (surface),adjac_lh)
	np.save("python_temp_med_%s/adjac_rh" % (surface),adjac_rh)

#step1
	x_covars = np.column_stack([np.ones(num_subjects),covars])
	y_lh = resid_covars(x_covars,data_lh)
	y_rh = resid_covars(x_covars,data_rh)
	del data_lh
	del data_rh
	merge_y = np.hstack((y_lh,y_rh))
	np.save("python_temp_med_%s/merge_y" % (surface),merge_y.astype(np.float32, order = "C"))
	del y_lh
	del y_rh
	n = len(merge_y)

#step2 mediation
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

#write TFCE images
	if not os.path.exists("output_med_%s" % surface):
		os.mkdir("output_med_%s" % surface)
	os.chdir("output_med_%s" % surface)

	write_vertStat_img('SobelZ',SobelZ[1,:num_vertex_lh],outdata_mask_lh, affine_mask_lh, surface, 'lh', bin_mask_lh, calcTFCE_lh, all_vertex)
	write_vertStat_img('SobelZ',SobelZ[1,num_vertex_lh:],outdata_mask_rh, affine_mask_rh, surface, 'rh', bin_mask_rh, calcTFCE_rh, all_vertex)
	write_vertStat_img('negSobelZ',(SobelZ[1,:num_vertex_lh]*-1),outdata_mask_lh, affine_mask_lh, surface, 'lh', bin_mask_lh, calcTFCE_lh, all_vertex)
	write_vertStat_img('negSobelZ',(SobelZ[1,num_vertex_lh:]*-1),outdata_mask_rh, affine_mask_rh, surface, 'rh', bin_mask_rh, calcTFCE_rh, all_vertex)
