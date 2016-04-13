#!/usr/bin/python

#    Various functions for Surf_tfce
#    Copyright (C) 2016  Tristram Lett, Lea Waller

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
from cython.cy_numstats import calc_beta_se

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

def write_vertStat_img(statname, vertStat, outdata_mask, affine_mask, surf, hemi, bin_mask, TFCEfunc, all_vertex):
	vertStat_out=np.zeros(all_vertex).astype(np.float32, order = "C")
	vertStat_out[bin_mask] = vertStat
	vertStat_TFCE = np.zeros_like(vertStat_out).astype(np.float32, order = "C")
	TFCEfunc.run(vertStat_out, vertStat_TFCE)
	outdata_mask[:,0,0] = vertStat_TFCE * (vertStat.max()/100)
	fsurfname = "%s_%s_%s_TFCE.mgh" % (statname,surf,hemi)
	os.system("echo %s_%s_%s,%f >> max_contrast_value.csv" % (statname,surf,hemi, outdata_mask[:,0,0].max()))
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)

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
