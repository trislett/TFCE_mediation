#!/usr/bin/env python

#    Various functions for TFCE_mediation
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

from .cynumstats import calc_beta_se

# Creation of adjacencty sets for TFCE connectivity
def create_adjac_vertex(vertices,faces): # basic version
	adjacency = [set([]) for i in xrange(vertices.shape[0])]
	for i in xrange(faces.shape[0]):
		adjacency[faces[i, 0]].add(faces[i, 1])
		adjacency[faces[i, 0]].add(faces[i, 2])
		adjacency[faces[i, 1]].add(faces[i, 0])
		adjacency[faces[i, 1]].add(faces[i, 2])
		adjacency[faces[i, 2]].add(faces[i, 0])
		adjacency[faces[i, 2]].add(faces[i, 1])
	return adjacency

def create_adjac_voxel (data_index,data_mask,num_voxel, dirtype=26): # default is 26 directions
	ind=np.where(data_index)
	dm=np.zeros_like(data_mask)
	x_dim,y_dim,z_dim=data_mask.shape
	adjacency = [set([]) for i in xrange(num_voxel)]
	label=0
	for x,y,z in zip(ind[0],ind[1],ind[2]):
		dm[x,y,z] = label
		label += 1
	for x,y,z in zip(ind[0],ind[1],ind[2]):
		xMin=max(x-1,0)
		xMax=min(x+1,x_dim-1)
		yMin=max(y-1,0)
		yMax=min(y+1,y_dim-1)
		zMin=max(z-1,0)
		zMax=min(z+1,z_dim-1)
		local_area = dm[xMin:xMax+1,yMin:yMax+1,zMin:zMax+1]
		if int(dirtype)==6:
			if local_area.shape!=(3,3,3): # check to prevent calculating adjacency at walls
				local_area = dm[x,y,z] 
			else:
				local_area = local_area * np.array([0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,1,0,0,0,0]).reshape(3,3,3)
		cV = int(dm[x,y,z])
		for j in local_area[local_area>0]:
			adjacency[cV].add(int(j))
	return adjacency

#writing statistics images

def write_vertStat_img(statname, vertStat, outdata_mask, affine_mask, surf, hemi, bin_mask, TFCEfunc, all_vertex):
	vertStat_out=np.zeros(all_vertex).astype(np.float32, order = "C")
	vertStat_out[bin_mask] = vertStat
	vertStat_TFCE = np.zeros_like(vertStat_out).astype(np.float32, order = "C")
	TFCEfunc.run(vertStat_out, vertStat_TFCE)
	outdata_mask[:,0,0] = vertStat_TFCE * (vertStat[np.isfinite(vertStat)].max()/100)
	fsurfname = "%s_%s_%s_TFCE.mgh" % (statname,surf,hemi)
	os.system("echo %s_%s_%s,%f >> max_TFCE_contrast_values.csv" % (statname,surf,hemi, outdata_mask[np.isfinite(outdata_mask[:,0,0])].max()))
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)
	outdata_mask[:,0,0] = vertStat_out
	fsurfname = "%s_%s_%s.mgh" % (statname,surf,hemi)
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)

def write_voxelStat_img(statname, voxelStat, out_path, data_index, affine, TFCEfunc):
	voxelStat_out = voxelStat.astype(np.float32, order = "C")
	voxelStat_TFCE = np.zeros_like(voxelStat_out).astype(np.float32, order = "C")
	TFCEfunc.run(voxelStat_out, voxelStat_TFCE)
	out_path[data_index] = voxelStat_TFCE * (voxelStat_out.max()/100)
	nib.save(nib.Nifti1Image(out_path,affine),"%s_TFCE.nii.gz" % (statname))
	os.system("echo %s,%f >> max_TFCE_contrast_values.csv" % (statname,out_path.max()))
	out_path[data_index] = voxelStat
	nib.save(nib.Nifti1Image(out_path,affine),"%s.nii.gz" % (statname))

#writing max TFCE values from permutations

def write_perm_maxTFCE_vertex(statname, vertStat, num_vertex, bin_mask_lh, bin_mask_rh, all_vertex,calcTFCE_lh,calcTFCE_rh):
	vertStat_out_lh=np.zeros(all_vertex).astype(np.float32, order = "C")
	vertStat_out_rh=np.zeros(all_vertex).astype(np.float32, order = "C")
	vertStat_TFCE_lh = np.zeros_like(vertStat_out_lh).astype(np.float32, order = "C")
	vertStat_TFCE_rh = np.zeros_like(vertStat_out_rh).astype(np.float32, order = "C")
	vertStat_out_lh[bin_mask_lh] = vertStat[:num_vertex]
	vertStat_out_rh[bin_mask_rh] = vertStat[num_vertex:]
	calcTFCE_lh.run(vertStat_out_lh, vertStat_TFCE_lh)
	calcTFCE_rh.run(vertStat_out_rh, vertStat_TFCE_rh)
	max_lh = vertStat_TFCE_lh[np.isfinite(vertStat_TFCE_lh)].max() * (vertStat_out_lh[np.isfinite(vertStat_out_lh)].max()/100)
	max_rh = vertStat_TFCE_rh[np.isfinite(vertStat_TFCE_rh)].max() * (vertStat_out_rh[np.isfinite(vertStat_out_rh)].max()/100)
	maxTFCE = np.array([max_lh,max_rh]).max()
	os.system("echo %.4f >> perm_%s_TFCE_maxVertex.csv" % (maxTFCE,statname))

def write_perm_maxTFCE_voxel(statname, voxelStat, TFCEfunc):
	voxelStat_out = voxelStat.astype(np.float32, order = "C")
	voxelStat_TFCE = np.zeros_like(voxelStat_out).astype(np.float32, order = "C")
	TFCEfunc.run(voxelStat_out, voxelStat_TFCE)
	maxval = voxelStat_TFCE.max() * (voxelStat_out.max()/100)
	os.system("echo %1.4f >> perm_%s_TFCE_maxVoxel.csv" % (maxval,statname))

#calculating Sobel Z statistics using T stats

def calc_sobelz(medtype, pred_x, depend_y, merge_y, n, num_vertex, alg = "aroian"):
	if medtype == 'I':
		PathA_beta, PathA_se = calc_beta_se(pred_x,merge_y,n,num_vertex)
		PathB_beta, PathB_se = calc_beta_se(np.column_stack([pred_x,depend_y]),merge_y,n,num_vertex)
		PathA_se = PathA_se[1]
		PathB_se = PathB_se[1]
	elif medtype == 'M':
		PathA_beta, PathA_se = calc_beta_se(pred_x,merge_y,n,num_vertex)
		PathB_beta, PathB_se = calc_beta_se(np.column_stack([depend_y,pred_x]),merge_y,n,num_vertex)
		PathA_se = PathA_se[1]
		PathB_se = PathB_se[1]
	elif medtype == 'Y':
		PathA_beta, _, _, _, PathA_se = linregress(pred_x, depend_y)
		PathB_beta, PathB_se = calc_beta_se(np.column_stack([depend_y,pred_x]),merge_y,n,num_vertex)
		PathB_se = PathB_se[1]
	else:
		print "Invalid mediation type"
		exit()
	ta = PathA_beta/PathA_se
	tb = PathB_beta/PathB_se
	if alg == 'aroian':
		#Aroian variant
		SobelZ = 1/np.sqrt((1/(tb**2))+(1/(ta**2))+(1/(ta**2*tb**2)))
	elif alg == 'sobel':
		#Sobel variant
		SobelZ = 1/np.sqrt((1/(tb**2))+(1/(ta**2)))
	elif alg == 'goodman':
		#Goodman variant
		SobelZ = 1/np.sqrt((1/(tb**2))+(1/(ta**2))-(1/(ta**2*tb**2)))
	else:
		print("Unknown indirect test algorithm")
		exit()
	return SobelZ
