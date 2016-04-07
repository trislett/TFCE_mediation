#!/usr/bin/python

#    Vertex-wise mediation with TFCE
#    Copyright (C) 2016  Tristram Lett

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import sys
import numpy as np
import nibabel as nib
from scipy.stats import linregress
from cython.cy_numstats import resid_covars,calc_beta_se
from cython.TFCE import Surf
from py_func import write_vertStat_img, create_adjac, calc_sobelz

if len(sys.argv) < 6:
	print "Usage: %s [predictor file] [covariate file] [dependent file] [surface (area or thickness)] [mediation type (M, Y, I)]" % (str(sys.argv[0]))
	print "Mediation types: M (neuroimage as mediator), Y (neuroimage as dependent), I (neuroimage as independent)"
	print "Optional arguments"
	print "Used supplied adjacency set ?mm: [1,2, or 3]"
	print "Custom adjacency set: [lh_adjacency_dist_?mm.npy] [rh_adjacency_dist_?mm.npy]"
else:
	cmdargs = str(sys.argv)
	scriptwd = os.path.dirname(os.path.realpath(sys.argv[0]))
	arg_predictor = str(sys.argv[1])
	arg_covars = str(sys.argv[2])
	arg_depend = str(sys.argv[3])
	surface = str(sys.argv[4])
	medtype = str(sys.argv[5])
	FWHM = '03B' # default 3mm smoothing

#load variables
	pred_x = np.genfromtxt(arg_predictor, delimiter=",")
	covars = np.genfromtxt(arg_covars, delimiter=",")
	depend_y = np.genfromtxt(arg_depend, delimiter=",")

#load data
	img_data_lh = nib.freesurfer.mghformat.load("lh.all.%s.%s.mgh" % (surface,FWHM))
	data_full_lh = img_data_lh.get_data()
	data_lh = np.squeeze(data_full_lh)
	affine_mask_lh = img_data_lh.get_affine()
	n = data_lh.shape[1]
	outdata_mask_lh = np.zeros_like(data_full_lh[:,:,:,1])
	img_data_rh = nib.freesurfer.mghformat.load("rh.all.%s.%s.mgh" % (surface,FWHM))
	data_full_rh = img_data_rh.get_data()
	data_rh = np.squeeze(data_full_rh)
	affine_mask_rh = img_data_rh.get_affine()
	outdata_mask_rh = np.zeros_like(data_full_rh[:,:,:,1])
	if not os.path.exists("lh.mean.%s.%s.mgh" % (surface,FWHM)):
		mean_lh = np.sum(data_lh,axis=1)/data_lh.shape[1]
		outmean_lh = np.zeros_like(data_full_lh[:,:,:,1])
		outmean_lh[:,0,0] = mean_lh
		nib.save(nib.freesurfer.mghformat.MGHImage(outmean_lh,affine_mask_lh),"lh.mean.%s.%s.mgh" % (surface,FWHM))
		mean_rh = np.sum(data_rh,axis=1)/data_rh.shape[1]
		outmean_rh = np.zeros_like(data_full_rh[:,:,:,1])
		outmean_rh[:,0,0] = mean_rh
		nib.save(nib.freesurfer.mghformat.MGHImage(outmean_rh,affine_mask_rh),"rh.mean.%s.%s.mgh" % (surface,FWHM))
	else:
		img_mean_lh = nib.freesurfer.mghformat.load("lh.mean.%s.%s.mgh" % (surface,FWHM))
		mean_full_lh = img_mean_lh.get_data()
		mean_lh = np.squeeze(mean_full_lh)
		img_mean_rh = nib.freesurfer.mghformat.load("rh.mean.%s.%s.mgh" % (surface,FWHM))
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
	elif len(sys.argv) == 7:
		print "Loading prior adjacency set for %s mm" % sys.argv[6]
		adjac_lh = np.load("%s/adjacency_sets/lh_adjacency_dist_%s.0_mm.npy" % (scriptwd,str(sys.argv[6])))
		adjac_rh = np.load("%s/adjacency_sets/rh_adjacency_dist_%s.0_mm.npy" % (scriptwd,str(sys.argv[6])))
	elif len(sys.argv) == 8:
		print "Loading prior adjacency set"
		arg_adjac_lh = str(sys.argv[6])
		arg_adjac_rh = str(sys.argv[7])
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
	np.save("python_temp_med_%s/num_subjects" % surface,n)
	np.save("python_temp_med_%s/num_vertex" % surface,num_vertex)
	np.save("python_temp_med_%s/num_vertex_lh" % (surface),num_vertex_lh)
	np.save("python_temp_med_%s/num_vertex_rh" % (surface),num_vertex_rh)
	np.save("python_temp_med_%s/all_vertex" % (surface),all_vertex)
	np.save("python_temp_med_%s/bin_mask_lh" % (surface),bin_mask_lh)
	np.save("python_temp_med_%s/bin_mask_rh" % (surface),bin_mask_rh)
	np.save("python_temp_med_%s/adjac_lh" % (surface),adjac_lh)
	np.save("python_temp_med_%s/adjac_rh" % (surface),adjac_rh)

#step1
	x_covars = np.column_stack([np.ones(n),covars])
	y_lh = resid_covars(x_covars,data_lh)
	y_rh = resid_covars(x_covars,data_rh)
	del data_lh
	del data_rh
	merge_y = np.hstack((y_lh,y_rh))
	np.save("python_temp_med_%s/merge_y" % (surface),merge_y.astype(np.float32, order = "C"))
	del y_lh
	del y_rh

#step2 mediation
	SobelZ = calc_sobelz(medtype, pred_x, depend_y, merge_y, n, num_vertex)

#write TFCE images
	if not os.path.exists("output_med_%s" % surface):
		os.mkdir("output_med_%s" % surface)
	os.chdir("output_med_%s" % surface)

	write_vertStat_img('SobelZ_%s' % (medtype),SobelZ[:num_vertex_lh],outdata_mask_lh, affine_mask_lh, surface, 'lh', bin_mask_lh, calcTFCE_lh, all_vertex)
	write_vertStat_img('SobelZ_%s' % (medtype),SobelZ[num_vertex_lh:],outdata_mask_rh, affine_mask_rh, surface, 'rh', bin_mask_rh, calcTFCE_rh, all_vertex)

