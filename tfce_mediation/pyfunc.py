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
import numpy as np
import nibabel as nib
import math
import sys
import struct
import uuid
from scipy.stats import linregress, t, f
from scipy.linalg import inv, sqrtm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from time import time

from tfce_mediation.cynumstats import calc_beta_se, cy_lin_lstsqr_mat, cy_lin_lstsqr_mat_residual, se_of_slope

# Creation of adjacencty sets for TFCE connectivity
def create_adjac_vertex(vertices,faces): # basic version
	adjacency = [set([]) for i in range(vertices.shape[0])]
	for i in range(faces.shape[0]):
		adjacency[faces[i, 0]].add(faces[i, 1])
		adjacency[faces[i, 0]].add(faces[i, 2])
		adjacency[faces[i, 1]].add(faces[i, 0])
		adjacency[faces[i, 1]].add(faces[i, 2])
		adjacency[faces[i, 2]].add(faces[i, 0])
		adjacency[faces[i, 2]].add(faces[i, 1])
	return adjacency

def create_adjac_voxel(data_index, data_mask, num_voxel, dirtype=26): # default is 26 directions
	ind = np.where(data_index)
	dm = np.zeros_like(data_mask)
	x_dim, y_dim, z_dim = data_mask.shape
	adjacency = [set([]) for i in range(num_voxel)]
	label = 0
	for x,y,z in zip(ind[0],ind[1],ind[2]):
		dm[x,y,z] = label
		label += 1
	for x,y,z in zip(ind[0],ind[1],ind[2]):
		xMin=max(x-1, 0)
		xMax=min(x+1, x_dim-1)
		yMin=max(y-1, 0)
		yMax=min(y+1, y_dim-1)
		zMin=max(z-1, 0)
		zMax=min(z+1, z_dim-1)
		local_area = dm[xMin:xMax+1,yMin:yMax+1,zMin:zMax+1]
		if int(dirtype)==6:
			if local_area.shape!=(3,3,3): # check to prevent calculating adjacency at walls
				local_area = dm[x,y,z]
			else:
				local_area = local_area * np.array([0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,1,0,0,0,0]).reshape(3,3,3)
		cV = int(dm[x,y,z])
		for j in local_area[local_area>0]:
			adjacency[cV].add(int(j))
	# convert to list
	adjacency = np.array([[x for x in sorted(i) if x != index] for index, i in enumerate(adjacency)]) # just convoluted enough
	adjacency[0] = []
	return adjacency

#writing statistics images

def write_vertStat_img(statname, vertStat, outdata_mask, affine_mask, surf, hemi, bin_mask, TFCEfunc, all_vertex, density_corr = 1, TFCE = True):
	vertStat_out=np.zeros(all_vertex).astype(np.float32, order = "C")
	vertStat_out[bin_mask] = vertStat
	if TFCE:
		vertStat_TFCE = np.zeros_like(vertStat_out).astype(np.float32, order = "C")
		TFCEfunc.run(vertStat_out, vertStat_TFCE)
		outdata_mask[:,0,0] = vertStat_TFCE * (vertStat[np.isfinite(vertStat)].max()/100) * density_corr
		fsurfname = "%s_%s_%s_TFCE.mgh" % (statname,surf,hemi)
		os.system("echo %s_%s_%s,%f >> max_TFCE_contrast_values.csv" % (statname,surf,hemi, outdata_mask[np.isfinite(outdata_mask[:,0,0])].max()))
		nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)
	outdata_mask[:,0,0] = vertStat_out
	fsurfname = "%s_%s_%s.mgh" % (statname,surf,hemi)
	nib.save(nib.freesurfer.mghformat.MGHImage(outdata_mask,affine_mask),fsurfname)

def write_voxelStat_img(statname, voxelStat, out_path, data_index, affine, TFCEfunc, imgext = '.nii.gz', TFCE = True):
	if TFCE:
		voxelStat_out = voxelStat.astype(np.float32, order = "C")
		voxelStat_TFCE = np.zeros_like(voxelStat_out).astype(np.float32, order = "C")
		TFCEfunc.run(voxelStat_out, voxelStat_TFCE)
		out_path[data_index] = voxelStat_TFCE * (voxelStat_out.max()/100)
		nib.save(nib.Nifti1Image(out_path,affine),"%s_TFCE%s" % (statname, imgext))
		os.system("echo %s,%f >> max_TFCE_contrast_values.csv" % (statname,out_path.max()))
	out_path[data_index] = voxelStat
	nib.save(nib.Nifti1Image(out_path,affine),"%s%s" % (statname, imgext))

#writing max TFCE values from permutations

def write_perm_maxTFCE_vertex(statname, vertStat, num_vertex, bin_mask_lh, bin_mask_rh, calcTFCE_lh,calcTFCE_rh, density_corr_lh = 1, density_corr_rh = 1):
	vertStat_out_lh=np.zeros(bin_mask_lh.shape[0]).astype(np.float32, order = "C")
	vertStat_out_rh=np.zeros(bin_mask_rh.shape[0]).astype(np.float32, order = "C")
	vertStat_TFCE_lh = np.zeros_like(vertStat_out_lh).astype(np.float32, order = "C")
	vertStat_TFCE_rh = np.zeros_like(vertStat_out_rh).astype(np.float32, order = "C")
	vertStat_out_lh[bin_mask_lh] = vertStat[:num_vertex]
	vertStat_out_rh[bin_mask_rh] = vertStat[num_vertex:]
	calcTFCE_lh.run(vertStat_out_lh, vertStat_TFCE_lh)
	calcTFCE_rh.run(vertStat_out_rh, vertStat_TFCE_rh)
	max_lh = vertStat_TFCE_lh[np.isfinite(vertStat_TFCE_lh)] * (vertStat_out_lh[np.isfinite(vertStat_out_lh)].max()/100) * density_corr_lh
	max_rh = vertStat_TFCE_rh[np.isfinite(vertStat_TFCE_rh)] * (vertStat_out_rh[np.isfinite(vertStat_out_rh)].max()/100) * density_corr_rh
	maxTFCE = np.array([max_lh.max(),max_rh.max()]).max()
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
		print("Invalid mediation type")
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

### tm_maths functions ###


def converter_try(vals):
	resline = []
	try:
		resline.append(int(vals))
		num_check=1
	except ValueError:
		try:
			resline.append(float(vals))
			num_check=1
		except ValueError:
			resline.append(vals)
			num_check=0
	return num_check

def loadnifti(imagename):
	if os.path.exists(imagename): # check if file exists
		if imagename.endswith('.nii.gz'):
			os.system("zcat %s > temp.nii" % imagename)
			img = nib.load('temp.nii')
			img_data = img.get_data()
			os.system("rm temp.nii")
		else:
			img = nib.load(imagename)
			img_data = img.get_data()
	else:
		print("Cannot find input image: %s" % imagename)
		exit()
	return (img,img_data)

def loadmgh(imagename):
	if os.path.exists(imagename): # check if file exists
		img = nib.freesurfer.mghformat.load(imagename)
		img_data = img.get_data()
	else:
		print("Cannot find input image: %s" % imagename)
		exit()
	return (img,img_data)

def loadtwomgh(imagename):
	if os.path.exists(imagename): # check if file exists
		lh_imagename = imagename
		rh_imagename = 'rh.%s' % (imagename.split('lh.',1)[1])
		if os.path.exists(rh_imagename): # check if file exists
			# truncated both hemispheres into a single array
			lh_img = nib.freesurfer.mghformat.load(lh_imagename)
			lh_img_data = lh_img.get_data()
			lh_mean_data = np.mean(np.abs(lh_img_data),axis=3)
			lh_mask_index = (lh_mean_data != 0)
			rh_img = nib.freesurfer.mghformat.load(rh_imagename)
			rh_img_data = rh_img.get_data()
			rh_mean_data = np.mean(np.abs(rh_img_data),axis=3)
			rh_mask_index = (rh_mean_data != 0)
			lh_img_data_trunc = lh_img_data[lh_mask_index]
			rh_img_data_trunc = rh_img_data[rh_mask_index]
			img_data_trunc = np.vstack((lh_img_data_trunc,rh_img_data_trunc))
			midpoint = lh_img_data_trunc.shape[0]
		else:
			print("Cannot find input image: %s" % rh_imagename)
			exit()
	else:
		print("Cannot find input image: %s" % imagename)
		exit()
	return (img_data_trunc, midpoint, lh_img, rh_img, lh_mask_index, rh_mask_index)

def savenifti(imgdata, img, index, imagename):
	outdata = imgdata.astype(np.float32, order = "C")
	if imgdata.ndim == 2:
		imgout = np.zeros((img.shape[0],img.shape[1],img.shape[2],outdata.shape[1]))
	elif imgdata.ndim == 1:
		imgout = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
	else:
		print('error')
	imgout[index]=outdata
	nib.save(nib.Nifti1Image(imgout.astype(np.float32, order = "C"),img.affine),imagename)

def savemgh(imgdata, img, index, imagename):
	outdata = imgdata.astype(np.float32, order = "C")
	if imgdata.ndim == 2:
		imgout = np.zeros((img.shape[0],img.shape[1],img.shape[2],outdata.shape[1]))
	elif imgdata.ndim == 1:
		imgout = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
	else:
		print('error')
	imgout[index]=outdata
	nib.save(nib.freesurfer.mghformat.MGHImage(imgout.astype(np.float32, order = "C"),img.affine),imagename)

#find nearest permuted TFCE max value that corresponse to family-wise error rate 
def find_nearest(array,value,p_array):
	idx = np.searchsorted(array, value, side="left")
	if idx == len(p_array):
		return p_array[idx-1]
	elif math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
		return p_array[idx-1]
	else:
		return p_array[idx]

# Z standardize or whiten
def zscaler(X, axis=0, w_mean=True, w_std=True):
	data = np.zeros_like(X)
	data[:] = np.copy(X)
	if w_mean:
		data -= np.mean(data, axis)
	if w_std:
		data /= np.std(data, axis)
	return data

# orthonormalizaiton with QR factorization
def gram_schmidt_orthonorm(X, columns=True):
	if columns:
		Q, _ = np.linalg.qr(X)
	else:
		Q, _ = np.linalg.qr(X.T)
		Q = Q.T
	return Q

#max-min standardization
def minmaxscaler(X, axis=0):
	X = (X - X.min(axis)) / (X.max(axis) - X.min(axis))
	return X

def PCAwhiten(X):
	from sklearn.decomposition import PCA
	pca = PCA(whiten=True)
	return (pca.fit_transform(X))

def ZCAwhiten(X):
	U, s, Vt = np.linalg.svd(X, full_matrices=False)
	return np.dot(U, Vt)

def orthog_columns(arr, normalize = True): # N x Exog
	arr = np.array(arr, dtype=np.float32)
	out_arr = []
	for column in range(arr.shape[1]):
		if normalize:
			X = stack_ones(zscaler(np.delete(arr,column,1)))
			y = zscaler(arr[:,column])
		else:
			X = stack_ones(np.delete(arr,column,1))
			y = arr[:,column]
		a = cy_lin_lstsqr_mat(X, y)
		out_arr.append(y - np.dot(X,a))
	return np.array(out_arr).T

def ortho_neareast(w):
	return w.dot(inv(sqrtm(w.T.dot(w))))

### surface conversion tools ###

def check_outname(outname):
	if os.path.exists(outname):
		outpath,outname = os.path.split(outname)
		if not outpath:
			outname = ("new_%s" % outname)
		else:
			outname = ("%s/new_%s" % (outpath,outname))
		print("Output file aleady exists. Renaming output file to %s" % outname)
		if os.path.exists(outname):
			print("%s also exists. Overwriting the file." % outname)
			os.remove(outname)
	return outname

def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1

# not used
def computeNormals(v, f):
	v_ = v[f]
	fn = np.cross(v_[:, 1] - v_[:, 0], v_[:, 2] - v_[:,0])
	fs = np.sqrt(np.sum((v_[:, [1, 2, 0], :] - v_) ** 2, axis = 2))
	fs_ = np.sum(fs, axis = 1) / 2 # heron's formula
	fa = np.sqrt(fs_ * (fs_ - fs[:, 0]) * (fs_ - fs[:, 1]) * (fs_ - fs[:, 2]))[:, None]
	vn = np.zeros_like(v, dtype = np.float32)
	vn[f[:, 0]] += fn * fa # weight by area
	vn[f[:, 1]] += fn * fa
	vn[f[:, 2]] += fn * fa
	vlen = np.sqrt(np.sum(vn ** 2, axis = 1))[np.any(vn != 0, axis = 1), None]
	vn[np.any(vn != 0, axis = 1), :] /= vlen
	return vn

def normalize_v3(arr):
	''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
	lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
	arr[:,0] /= lens
	arr[:,1] /= lens
	arr[:,2] /= lens
	return arr

# Y = Categorical Dependent variable
# X = Neuroimage
# covars = covariates
# scale = minmaxscaling
# X_output = output the (scaled) neuroimage residuals
#def chi_sqr_test(Y, X, covars = None, scale = True, X_output = False):
#	from sklearn.feature_selection import chi2
#	from scipy.stats import norm
#	from sklearn import preprocessing

#	if not all(np.equal(item, int(item)) for item in Y):
#		print "Y must contain interger categorical variables"
#		quit()
#	if covars is not None:
#		X = resid_covars(np.column_stack([np.ones(len(covars)),covars]), X)
#	else:
#		if len(Y) == X.shape[1]:
#			X = X.T
#	if scale:
#		min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#		X = min_max_scaler.fit_transform(X)
#	Chi, P = chi2(X, Y)
#	Z = norm.ppf(1-P)
#	if X_output:
#		return Chi, Z, P, X
#	else:
#		return Chi, Z, P

# input functions
def convert_mni_object(obj_file):
	# adapted from Jon Pipitone's script https://gist.github.com/pipitone/8687804
	obj = open(obj_file)
	_, _, _, _, _, _, numpoints = obj.readline().strip().split()
	numpoints = int(numpoints)
	vertices=[]
	normals=[]
	triangles=[]

	for i in range(numpoints):
		x, y, z = list(map(float,obj.readline().strip().split())) 
		vertices.append((x, y, z))
	assert obj.readline().strip() == ""
	# numpoints normals as (x,y,z)
	for i in range(numpoints):
		normals.append(tuple(map(float,obj.readline().strip().split())))

	assert obj.readline().strip() == ""
	nt=int(obj.readline().strip().split()[0]) # number of triangles
	_, _, _, _, _ = obj.readline().strip().split()
	assert obj.readline().strip() == ""
	# rest of the file is a list of numbers
	points = list(map(int, "".join(obj.readlines()).strip().split()))
	points = points[nt:]	# ignore these.. (whatever they are)
	for i in range(nt): 
		triangles.append((points.pop(0), points.pop(0), points.pop(0)))
	return np.array(vertices), np.array(triangles)

def convert_fs(fs_surface):
	v, f = nib.freesurfer.read_geometry(fs_surface)
	return v, f

def convert_gifti(gifti_surface):
	img = nib.load(gifti_surface)
	v, f = img.darrays[0].data, img.darrays[1].data
	return v, f

def convert_ply(name_ply):
	element = []
	size = []
	vertex_info = []
	vertex_dtype = []
	face_dtype = []
	face_info = []
	vertex_property = 0
	face_property = 0
	ply_ascii = False

	obj = open(name_ply)
	reader = obj.readline().strip().split()
	firstword = reader[0]

	# READ HEADER
	while firstword != 'end_header':
		reader = obj.readline().strip().split()
		firstword = reader[0]
		if firstword == 'format':
			ply_format = reader[1]
			if ply_format == 'binary_little_endian':
				ply_format = '<'
			elif ply_format == 'binary_big_endian':
				ply_format = '>'
			else:
				ply_ascii = True
		if firstword == 'element':
			element.append((reader[1]))
			size.append((reader[2]))
			if reader[1] == 'vertex':
				vertex_property = 1
			else:
				vertex_property = 0
			if reader[1] == 'face':
				face_property = 1
			else:
				face_property = 0
		if reader[0] == 'property':
			if vertex_property == 1:
				vertex_dtype.append((reader[1]))
				vertex_info.append((reader[2]))
			elif face_property == 1:
				face_dtype.append((reader[2]))
				face_dtype.append((reader[3]))
				face_info.append((reader[4]))
			else:
				print("Unknown property")

	# READ ELEMENTS
	for e in range(len(element)):
		# VERTEX DATA
		if element[e] == 'vertex':
			v = np.zeros((int(size[e]), 3), dtype=np.float32)
			c = np.zeros((int(size[e]), 3), dtype=np.uint8)
			if ply_ascii:
				for i in range(int(size[e])):
					reader = obj.readline().strip().split()
					v[i, 0] = np.array(reader[0]).astype(np.float)
					v[i, 1] = np.array(reader[1]).astype(np.float)
					v[i, 2] = np.array(reader[2]).astype(np.float)
					if len(vertex_info) == 6:
						c[i, 0] = np.array(reader[3]).astype(np.uint8)
						c[i, 1] = np.array(reader[4]).astype(np.uint8)
						c[i, 2] = np.array(reader[5]).astype(np.uint8)
			else:
				struct_fmt = ply_format
				for i in range(len(vertex_dtype)):
					if vertex_dtype[i] == 'float':
						struct_fmt += 'f'
					if vertex_dtype[i] == 'uchar':
						struct_fmt += 'B'
					if vertex_dtype[i] == 'int':
						struct_fmt += 'i'
				struct_len = struct.calcsize(struct_fmt)
				struct_unpack = struct.Struct(struct_fmt).unpack_from
				vcounter = 0
				while vcounter != int(size[e]):
					if len(vertex_dtype) > 3:
						s = struct_unpack(obj.read(struct_len))
						v[vcounter] = s[:3]
						c[vcounter] = s[3:]
						vcounter += 1
					else:
						s = struct_unpack(obj.read(struct_len))
						v[vcounter] = s[:3]
						vcounter += 1
		# FACE DATA
		if element[e] == 'face':
			if ply_ascii:
				reader = obj.readline().strip().split()
				numf = int(reader[0])
				f = np.zeros((int(size[e]), numf), dtype=int32)
				f[0] = reader[1:]
				fcounter = 1
				while fcounter != int(size[e]):
					reader = obj.readline().strip().split()
					f[fcounter] = reader[1:]
					fcounter += 1
			else:
				if face_dtype[0] == 'uchar':
					fcounter = 0
					while fcounter != int(size[e]):
						struct_unpack = struct.Struct(ply_format + 'B').unpack_from
						numf = struct_unpack(obj.read(1))[0]
						# creates empty face array if it doesn't exists
						try:
							f
						except NameError:
							f = np.zeros((int(size[e]), numf), dtype=int32)
						struct_fmt = ply_format
						for i in range(int(numf)):
							struct_fmt += 'i'
						struct_len = struct.calcsize(struct_fmt)
						struct_unpack = struct.Struct(struct_fmt).unpack_from
						s = struct_unpack(obj.read(struct_len))
						f[fcounter] = s
						fcounter += 1
	return (v, f, c)

def convert_fslabel(name_fslabel):
	obj = open(name_fslabel)
	reader = obj.readline().strip().split()
	reader = np.array(obj.readline().strip().split())
	if reader.ndim == 1:
		num_vertex = reader[0].astype(int)
	else:
		print('Error reading header')
	v_id = np.zeros((num_vertex)).astype(int)
	v_ras = np.zeros((num_vertex,3)).astype(np.float)
	v_value = np.zeros((num_vertex)).astype(np.float)
	for i in range(num_vertex):
		reader = obj.readline().strip().split()
		v_id[i] = np.array(reader[0]).astype(int)
		v_ras[i] = np.array(reader[1:4]).astype(np.float)
		v_value[i] = np.array(reader[4]).astype(np.float)
	return (v_id, v_ras, v_value)

#output functions

def save_waveform(v,f, outname):
	if not outname.endswith('obj'):
		outname += '.obj'
	outname=check_outname(outname)
	with open(outname, "a") as o:
		for i in range(len(v)):
			o.write("v %1.6f %1.6f %1.6f\n" % (v[i,0],v[i,1], v[i,2]) )
		for j in range(len(f)):
			o.write("f %d %d %d\n" % (f[j,0],f[j,1], f[j,2]) )
		o.close()

def save_stl(v,f, outname):
	if not outname.endswith('stl'):
		outname += '.stl'
	outname=check_outname(outname)
	v = np.array(v, dtype=np.float32, order = "C")
	f = np.array(f, dtype=int32, order = "C")
	tris = v[f]
	n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
	n = normalize_v3(n)
	with open(outname, "a") as o:
		o.write("solid surface\n")
		for i in range(tris.shape[0]):
			o.write("facet normal %1.6f %1.6f %1.6f\n"% (n[i,0],n[i,0],n[i,0]))
			o.write("outer loop\n")
			o.write("vertex %1.6f %1.6f %1.6f\n" % (tris[i,0,0],tris[i,0,1],tris[i,0,2]))
			o.write("vertex %1.6f %1.6f %1.6f\n" % (tris[i,1,0],tris[i,1,1],tris[i,1,2]))
			o.write("vertex %1.6f %1.6f %1.6f\n" % (tris[i,2,0],tris[i,2,1],tris[i,2,2]))
			o.write("endloop\n")
			o.write("endfacet\n")
		o.write("endfacet\n")
		o.close()


def save_fs(v,f, outname):
	if not outname.endswith('srf'):
		outname += '.srf'
	outname=check_outname(outname)
	nib.freesurfer.io.write_geometry(outname, v, f)


def save_ply(v, f, outname, color_array=None, output_binary=True):
	# check file extension
	if not outname.endswith('ply'):
		if output_binary:
			outname += '.ply'
		else:
			outname += '.ascii.ply'
	outname=check_outname(outname)

	# write header 
	header = ("ply\n")
	if output_binary:
		header += ("format binary_%s_endian 1.0\n" % (sys.byteorder))
		if sys.byteorder == 'little':
			output_fmt = '<'
		else:
			output_fmt = '>'
	else:
		header += ("format ascii 1.0\n")
	header += ("comment made with TFCE_mediation\n")
	header += ("element vertex %d\n" % len(v))
	header += ("property float x\n")
	header += ("property float y\n")
	header += ("property float z\n")
	if color_array is not None:
		header += ("property uchar red\n")
		header += ("property uchar green\n")
		header += ("property uchar blue\n")
	header += ("element face %d\n" % len(f))
	header += ("property list uchar int vertex_index\n")
	header += ("end_header\n")

	# write to file
	if output_binary:
		with open(outname, "a") as o:
			o.write(header)
			o.close()
		with open(outname, "ab") as o:
			for i in range(len(v)):
					if color_array is not None:
						o.write(
							struct.pack(output_fmt + 'fffBBB', v[i, 0], v[i, 1], v[i, 2], color_array[i, 0], color_array[i, 1], color_array[i, 2]))
					else:
						o.write(struct.pack(output_fmt + 'fff', v[i, 0], v[i, 1], v[i, 2]))
			for j in range(len(f)):
				o.write(struct.pack('<Biii', 3, f[j, 0], f[j, 1], f[j, 2]))
	else:
		with open(outname, "a") as o:
			for i in range(len(v)):
				if color_array is not None:
					o.write("%1.6f %1.6f %1.6f %d %d %d\n" % (v[i, 0], v[i, 1], v[i, 2], color_array[i, 0], color_array[i, 1], color_array[i, 2]))
				else:
					o.write("%1.6f %1.6f %1.6f\n" % (v[i, 0], v[i, 1], v[i, 2]))
			for j in range(len(f)):
				o.write("3 %d %d %d\n" % (f[j, 0], f[j, 1], f[j, 2]))


#vertex paint functions
def convert_redtoyellow(threshold,img_data, baseColour=[227,218,201], save_colorbar = True):
	color_array = np.zeros((img_data.shape[0],3))
	color_cutoffs = np.linspace(threshold[0],threshold[1],256)
	colored_img_data = np.zeros_like(img_data)
	cV=0
	for k in img_data:
		colored_img_data[cV] = np.searchsorted(color_cutoffs, k, side="left")
		cV+=1
	color_array[:,0]=255
	color_array[:,1]=np.copy(colored_img_data)
	color_array[img_data<threshold[0]] = baseColour
	color_array[img_data>threshold[1]] = [255,255,0]

	cmap_name = 'red_yellow'
	cmap_array = np.array(( (np.ones(256)*255), np.linspace(0,255,256), np.zeros(256))).T
	rl_cmap = colors.ListedColormap(cmap_array/255)
	if save_colorbar:
		write_colorbar(threshold, rl_cmap, cmap_name, 'png')
		plt.clf()
	return color_array

def convert_bluetolightblue(threshold, img_data, baseColour=[227,218,201], save_colorbar = True):
	color_array = np.zeros((img_data.shape[0],3))
	color_cutoffs = np.linspace(threshold[0],threshold[1],256)
	colored_img_data = np.zeros_like(img_data)
	cV=0
	for k in img_data:
		colored_img_data[cV] = np.searchsorted(color_cutoffs, k, side="left")
		cV+=1
	color_array[:,1]=np.copy(colored_img_data)
	color_array[:,2]=255
	color_array[img_data<threshold[0]] = baseColour
	color_array[img_data>threshold[1]] = [0,255,255]

	cmap_name = 'blue_lightblue'
	cmap_array = np.array(( np.zeros(256), np.linspace(0,255,256), (np.ones(256)*255))).T
	blb_cmap = colors.ListedColormap(cmap_array/255)
	if save_colorbar:
		write_colorbar(threshold, blb_cmap, cmap_name, 'png')
		plt.clf()
	return color_array

def convert_mpl_colormaps(threshold,img_data, cmapName, baseColour=[227,218,201], save_colorbar = True):
	cmapFunc = plt.get_cmap(str(cmapName))
	color_array = np.zeros((img_data.shape[0],3))
	color_cutoffs = np.linspace(threshold[0],threshold[1],256)
	cV=0
	for k in img_data:
		temp_ = np.array(cmapFunc(np.searchsorted(color_cutoffs, k, side="left")))*255
		color_array[cV,:] = ((np.around(temp_[0]), np.around(temp_[1]), np.around(temp_[2])))
		cV+=1
	color_array[img_data<threshold[0]] = baseColour
	temp_ = np.array(cmapFunc(np.searchsorted(color_cutoffs, color_cutoffs[255], side="left")))*255 # safer
	color_array[img_data>=threshold[1]] = ((int(temp_[0]), int(temp_[1]), int(temp_[2])))
	if save_colorbar:
		write_colorbar(threshold, cmapFunc, cmapName, 'png')
		plt.clf()
	return color_array

def convert_fsannot(annot_name):
	labels, ctab, names  = nib.freesurfer.read_annot(annot_name)
	labels[labels==-1] = 0 # why is this necessary?????
	color_array = ctab[labels]
	color_array = color_array[:,:3]
	write_annot_legend(ctab, names, annot_name)
	return color_array

#write legends
def write_annot_legend(ctab, names, annot_name, outtype = 'png'):
	all_patches = []
	for i in range(len(names)):
		all_patches.append(mpatches.Patch(color=(ctab[i][:3]/256), label=names[i]))
	plt.figure()
	plt.legend(handles=[all_patches[l] for l in range(len(names))], loc=2)
	plt.axis('off')
	plt.savefig("%s_legend.%s" % (os.path.basename(annot_name), outtype),bbox_inches='tight')

def write_colorbar(threshold, input_cmap, name_cmap, outtype = 'png'):
	a = np.array([[threshold[0],threshold[1]]])
	plt.figure()
	plt.imshow(a, cmap=input_cmap)
	plt.gca().set_visible(False)
	cax = plt.axes([0.1, 0.1, 0.03, 0.8])
	plt.colorbar(orientation="vertical", cax=cax)
	plt.savefig("%s_colorbar.%s" % (os.path.basename(name_cmap), outtype),bbox_inches='tight')
	plt.clf()

def convert_voxel(img_data, affine = None, threshold = None, data_mask = None, absthreshold = None):
	"""
	Converts a voxel image to a surface including outputs voxel values to paint vertex surface.
	
	Parameters
	----------
	img_data : array
		image array
	affine : array
		 affine [4x4] to convert vertices values to native space (Default = None)
	data_mask : array
		use a mask to create a surface backbone (Default = None)
	threshold : float
		threshold for output of voxels (Default = None)
	absthreshold : float
		threshold for output of abs(voxels) (Default = None)
		
	Returns
	-------
		v : array
			vertices
		f : array
			faces
		values : array
			scalar values
	
	"""
	try:
		from skimage import measure
	except:
		print("Error skimage is required")
		quit()

	if threshold is not None:
		print("Zeroing data less than threshold = %1.2f" % threshold)
		img_data[img_data<threshold] = 0
	if absthreshold is not None:
		print("Zeroing absolute values less than threshold = %1.2f" % absthreshold)
		img_data[np.abs(img_data)<absthreshold] = 0
	if data_mask is not None:
		print("Including mask")
		data_mask *= .1
		data_mask[img_data!=0] = img_data[img_data!=0]
		del img_data
		img_data = np.copy(data_mask)
	try:
		v, f, _, values = measure.marching_cubes(img_data)
		if affine is not None:
			print("Applying affine transformation")
			v = nib.affines.apply_affine(affine,v)
	except:
		print("No voxels above threshold")
		v = f = values = []
	return v, f, values

# Check if okay to remove
def basic_laplacian_smoothing(v, f, adjacency = None, scalar = None):
	"""
	Depreciated.
	"""
	n = v.shape[0]
	laplace_v = np.empty((n,3))

	if adjacency is None:
		adjacency = create_adjac_vertex(v,f)
	if scalar is not None:
		values = np.empty((n))

	for i in range(n):
		neighbors = list(adjacency[i])
		if len(neighbors) > 0:
			laplace_v[i] = np.mean([v[j] for j in neighbors], axis = 0)
			if scalar is not None:
				values[i] = np.mean([scalar[j] for j in neighbors])
	if scalar is not None:
		return (laplace_v, f, values)
	else:
		return (laplace_v, f)

# Check if okay to remove
def surface_smooth(v, f, adjacency, iter_num = 0, scalar = None, lambda_w = 1.0, mode = 'laplacian', v_weighted = True):
	"""
	Depreciated.
	"""
	k = 0.1
	mu_w = -lambda_w/(1-k*lambda_w)
	n = v.shape[0]
	v_smooth = np.empty((n,3))

	if scalar is not None:
		values = np.empty((n))

	def v_new(v, vneighbors, v_factor, sneighbors = None):
		if len(vneighbors) > 0:
			weights = np.power(np.linalg.norm((vneighbors - v), axis = 1)[:,None], -1)
			vectors = weights * vneighbors
			w_vertex = v + (v_factor * ((np.sum(vectors, axis = 0)/np.sum(weights)) - v))
			if sneighbors is not None:
				w_scalar = np.mean(np.sum(weights * sneighbors,axis = 0) / np.sum(weights))
				return w_vertex, w_scalar
			return w_vertex

	def v_new_unweighted(v, vneighbors, v_factor = None, sneighbors = None):
		if sneighbors is not None:
			return np.mean(vneighbors, axis = 0), np.mean(sneighbors)
		else:
			return np.mean(vneighbors, axis = 0)

	if v_weighted:
		calc_Vnew = v_new
	else:
		calc_Vnew = v_new_unweighted


	for i in range(n):
		neighbors = list(adjacency[i])
		if len(neighbors) > 0:

			if iter_num % 2 == 0:
				if i == 1:
					print("Smoothing iteration (positive factor): %d" % (iter_num+1))
				if scalar is not None:
					v_smooth[i], values[i] = calc_Vnew(v[i], v[neighbors], lambda_w, scalar[neighbors])
				else:
					v_smooth[i] = calc_Vnew(v[i], v[neighbors], lambda_w)
			else:
				if mode == 'taubin':
					if i == 1:
						print("Smoothing iteration (negative factor): %d" % (iter_num+1))

					if scalar is not None:
						v_smooth[i], values[i] = calc_Vnew(v[i], v[neighbors], mu_w, scalar[neighbors])
					else:
						v_smooth[i] = calc_Vnew(v[i], v[neighbors], mu_w)


				elif mode == 'laplacian':
					if i == 1:
						print("Smoothing iteration (positive factor): %d" % (iter_num+1))

					if scalar is not None:
						v_smooth[i], values[i] = calc_Vnew(v[i], v[neighbors], lambda_w, scalar[neighbors])
					else:
						v_smooth[i] = calc_Vnew(v[i], v[neighbors], lambda_w)

				else:
					print("Error: smoothing type not understood")

	if scalar is not None:
		return (v_smooth, f, values)
	else:
		return (v_smooth, f)


def vectorized_surface_smooth(v, f, adjacency, number_of_iter = 5, scalar = None, lambda_w = 0.5, mode = 'laplacian', weighted = True):
	"""
	Applies Laplacian (Gaussian) or Taubin (low-pass) smoothing with option to smooth single volume
	
	Citations
	----------
	
	Herrmann, Leonard R. (1976), "Laplacian-isoparametric grid generation scheme", Journal of the Engineering Mechanics Division, 102 (5): 749-756.
	Taubin, Gabriel. "A signal processing approach to fair surface design." Proceedings of the 22nd annual conference on Computer graphics and interactive techniques. ACM, 1995.
	
	
	Parameters
	----------
	v : array
		vertex array
	f : array
		face array
	adjacency : array
		adjacency array

	
	Flags
	----------
	number_of_iter : int
		number of smoothing iterations
	scalar : array
		apply the same smoothing to a image scalar
	lambda_w : float
		lamda weighting of degree of movement for each iteration
		The weighting should never be above 1.0
	mode : string
		The type of smoothing can either be laplacian (which cause surface shrinkage) or taubin (no shrinkage)
		
	Returns
	-------
	v_smooth : array
		smoothed vertices array
	f : array
		f = face array (unchanged)
	
	Optional returns
	-------
	values : array
		smoothed scalar array
	
	"""
	k = 0.1
	mu_w = -lambda_w/(1-k*lambda_w)

	lengths = np.array([len(a) for a in adjacency])
	maxlen = max(lengths)
	padded = [list(a) + [-1] * (maxlen - len(a)) for a in adjacency]
	adj = np.array(padded)
	w = np.ones(adj.shape, dtype=float)
	w[adj<0] = 0.
	val = (adj>=0).sum(-1).reshape(-1, 1)
	w /= val
	w = w.reshape(adj.shape[0], adj.shape[1],1)

	vorig = np.zeros_like(v)
	vorig[:] = v
	if scalar is not None:
		scalar[np.isnan(scalar)] = 0
		sorig = np.zeros_like(scalar)
		sorig[:] = scalar

	for iter_num in range(number_of_iter):
		if weighted:
			vadj = v[adj]
			vadj = np.swapaxes(v[adj],1,2)
			weights = np.zeros((v.shape[0], maxlen))
			for col in range(maxlen):
				weights[:,col] = np.power(np.linalg.norm(vadj[:,:,col] - v, axis=1),-1)
			weights[adj==-1] = 0
			vectors = np.einsum('abc,adc->acd', weights[:,None], vadj)

			if scalar is not None:
				scalar[np.isnan(scalar)] = 0

				sadj = scalar[adj]
				sadj[adj==-1] = 0
				if lambda_w < 1:
					scalar = (scalar*(1-lambda_w)) + lambda_w*(np.sum(np.multiply(weights, sadj),axis=1) / np.sum(weights, axis = 1))
				else:
					scalar = np.sum(np.multiply(weights, sadj),axis=1) / np.sum(weights, axis = 1)
				scalar[np.isnan(scalar)] = sorig[np.isnan(scalar)] # hacky scalar nan fix
			if iter_num % 2 == 0:
				v += lambda_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			elif mode == 'taubin':
				v += mu_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			elif mode == 'laplacian':
				v += lambda_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			else:
				print("Error: mode %s not understood" % mode)
				quit()
			v[np.isnan(v)] = vorig[np.isnan(v)] # hacky vertex nan fix
		else:
			if scalar is not None:
				sadj = scalar[adj]
				sadj[adj==-1] = 0

				if lambda_w < 1:
					scalar = (scalar*(1-lambda_w)) + (lambda_w*np.divide(np.sum(sadj, axis = 1),lengths))
				else:
					scalar = np.divide(np.sum(sadj, axis = 1),lengths)
			if iter_num % 2 == 0:
				v += np.array(lambda_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			elif mode == 'taubin':
				v += np.array(mu_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			elif mode == 'laplacian':
				v += np.array(lambda_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			else:
				print("Error: mode %s not understood" % mode)
				quit()

	if scalar is not None:
		return (v, f, scalar)
	else:
		return (v, f)


def image_regression(y, image_x, pred_x, covars = None, normalize = False, verbose = True):
	"""
	Applies regression using a voxel/vertex wise regressor
	
	Parameters
	----------
	y : array
		masked image data (V x Subjects)
	image_x : array
		image regressor
	pred_x : array
		predictors

	
	Flags
	----------
	covars : array
		covariates
	normalize : bool
		Z-scale the image regressor
	verbose : bool
		Verbose output
	
	Returns
	-------
	Tval : array
		T-values image of the predictors
	Timg : array
		T-value of image regressor
	
	"""
	start_time = time()
	nv = y.shape[0]
	n = y.shape[1]
	if np.all(covars != None):
		regressors = np.column_stack((pred_x, covars))
	else:
		regressors = pred_x
	regressors = np.column_stack([np.ones(len(regressors)),regressors])
	arr = np.zeros((nv,len(regressors.T)+1))

	if normalize:
		image_x = zscaler(image_x, axis=0, w_mean=True, w_std=True)

	for i in range(nv):
		if i % 5000 == 0:
			print(i)
		if (image_x[i,:].std() < 0.01) or (np.any(np.isnan(image_x[i,:]))):
			arr[i,:] = 0
		else:
			temp_data = image_x[i,:]
			temp_y =  y[i,:]
			X = np.column_stack((regressors, temp_data))
			k = len(X.T)
			invXX = np.linalg.inv(np.dot(X.T, X))
			a = cy_lin_lstsqr_mat(X, temp_y)
			sigma2 = np.sum((temp_y - np.dot(X,a))**2,axis=0) / (n - k)
			se = np.sqrt(np.diag(sigma2 * invXX))
			arr[i] = a / se
	print(("Finished. Image-wise independent variable regression took %.1f seconds" % (time() - start_time)))
	Tval = np.array(arr[:,:len(pred_x.T)+1], dtype=np.float32)
	Timg = np.array(np.squeeze(arr[:,-1:]), dtype=np.float32)
	return(Tval, Timg)

def image_reg_VIF(y, regressors):
	"""
	Calculates the variance inflation factor (VIF) to test for multicollinearity
	VIF > 10 is considered high multicollinearity across an image
	
	Parameters
	----------
	y : array
		masked neuroimage (endogenous) data array
	regressors : array
		exogenous variable array (without intercept) 
	
	Returns
	-------
	VIF : array
		The variance inflation factor across the image.
	
	"""
	X = np.column_stack([np.ones(len(regressors)),regressors])
	a = cy_lin_lstsqr_mat(X, y)
	resids = y - np.dot(X,a)
	RSS = np.sum(resids**2,axis=0)
	TSS = np.sum((y - np.mean(y, axis =0))**2, axis = 0)
	R2 = 1 - (RSS/TSS)
	VIF = 1 / (1-R2)
	VIF[np.isnan(VIF)] = 0
	return VIF

def erode_3D_image(img_data, erode_iter = 2, do_binary_opening = False, do_binary_closing = False, do_remove_small_holes = False):
	"""
	Applies erosion of 3D voxel images using skimage.morphology

	Parameters
	----------

	img_data : array
		image data (X,Y,Z)
	erode_iter : int
		number of iterations of erosion
	
	Flags
	----------
	
	do_binary_opening : bool
		erosion followed by a dilation (default = False)
	do_binary_closing : bool
		dilation followed by an erosion (default = False)
	do_remove_small_holes : bool
		Remove continguous holes smaller than the specified size (default = False)
	
	Returns
	-------
	img_data : array
		eroded image
	erode_mask : array
		eroded image mask
	
	"""
	from skimage.morphology import binary_erosion, binary_opening, binary_closing, remove_small_holes
	mask_data = np.zeros_like(img_data)
	mask_data[:] = img_data
	mask_data[mask_data!=0]=1
	erode_mask = np.zeros_like(img_data)
	xdim, ydim, zdim = mask_data.shape

	for i in range(erode_iter):
		for x in range(xdim):
			img_slice = mask_data[x,:,:]
			if img_slice.mean() !=0:
				if do_binary_opening:
					img_slice = binary_opening(img_slice)*1
				if do_binary_closing:
					img_slice = binary_closing(img_slice)*1
				if do_remove_small_holes:
					img_slice = remove_small_holes(img_slice, 2)*1
				img_slice = binary_erosion(img_slice)*1
			else:
				img_slice = mask_data[x,:,:]
			erode_mask[x,:,:] = img_slice
		for y in range(ydim):
			img_slice = mask_data[:,y,:]
			if img_slice.mean() !=0:
				if do_binary_opening:
					img_slice = binary_opening(img_slice)*1
				if do_binary_closing:
					img_slice = binary_closing(img_slice)*1
				if do_remove_small_holes:
					img_slice = remove_small_holes(img_slice, 2)*1
				img_slice = binary_erosion(img_slice)*1
			else:
				img_slice = mask_data[:,y,:]
			erode_mask[:,y,:] = img_slice
		for z in range(zdim):
			img_slice = mask_data[:,:,z]
			if img_slice.mean() !=0:
				if do_binary_opening:
					img_slice = binary_opening(img_slice)*1
				if do_binary_closing:
					img_slice = binary_closing(img_slice)*1
				if do_remove_small_holes:
					img_slice = remove_small_holes(img_slice, 2)*1
				img_slice = binary_erosion(img_slice)*1
			else:
				img_slice = mask_data[:,:,z]
			erode_mask[:,:,z] = img_slice
		mask_data[:] = erode_mask
	img_data *= erode_mask
	return img_data, erode_mask

def import_voxel_neuroimage(image_path, mask_index = None):
	"""
	Low-RAM voxel-image importer using nibabel
	
	Parameters
	----------
	image_path : string
		PATH/TO/IMAGE
	
	Flags
	----------
	mask_index : index array (bool or binary)
		Masks the image by non-zero index (Default = None)
	
	Returns
	-------
	image : array
		nibable image object
	
	OR
	
	image_data : array
		numpy array of masked image data
	
	"""
	if not os.path.exists(image_path):
		print('Error: %s not found' % image_path)
		quit()
	base, file_ext = os.path.splitext(image_path)
	if file_ext == '.gz':
		file_ext = os.path.splitext(base)[1]
		if file_ext == '.nii':
			if os.path.getsize(image_path) < 50000000:
				image = nib.load(image_path)
				image_data = image.get_data()
			else:
				tempname = str(uuid.uuid4().hex) + '.nii'
				os.system("zcat %s > %s" % (image_path,tempname))
				image = nib.load(tempname)
				image_data = image.get_data()
				os.system("rm %s" % tempname)
		else:
			print('Error: filetype for %s is not supported' % image_path)
	elif file_ext == '.nii':
		image = nib.load(image_path)
		image_data = image.get_data()
	elif file_ext == '.mnc':
		image = nib.load(image_path)
		image_data = image.get_data()
	else:
		print('Error filetype for %s is not supported' % file_ext)
		quit()
	print("Imported:\t%s" % image_path)
	if mask_index is not None:
		image_data = image_data[mask_index]
		return image_data
	else:
		return image, image_data

def rm_anova(data, output_sig = False, output_eta_sq = False):
	"""
	Repeated measure ANOVA for longitudinal dependent variables
	
	Parameters
	----------
	data : array
		Data array (N_intervals, N_individuals, N_dependent variables)
	
	Optional Flags
	----------
	output_sig : bool
		outputs p-values of F-statistics
	
	Returns
	-------
	F : array
		F-statistics of the interval variable
	P : array
		P-statistics of the interval variable (P = None if output_sig = False)
	partial_eta_sq : array
		Partial eta squared of the interval variable (partial_eta_sq = None if output_eta_sq = False)
	"""

	k = data.shape[0]
	ni = data.shape[1]
	mu_grand = np.zeros_like(np.mean(data[0],0))
	for i in range(k):
		mu_grand += np.mean(data[i],0)
	mu_grand = np.divide(mu_grand,k)

	SStime = np.zeros_like(np.mean(data[0],0))
	for i in range(k):
		SStime += (np.mean(data[i],0)-mu_grand)**2
	SStime *= ni

	SSw = np.zeros_like(np.mean(data[0],0))
	for i in range(k):
		SSw += np.sum(np.square(data[i] - np.mean(data[i],0)),0)

	SSsub = k * np.sum(np.square(np.divide((data.sum(0)),k) - mu_grand),0)
	SSerror = SSw - SSsub

	df_time = (k-1)
	df_error = (ni -1) * df_time

	MStime = np.divide(SStime, df_time)
	MSerror = np.divide(SSerror, df_error)
	F = np.divide(MStime, MSerror)
	if output_sig:
		P = 1 - f.cdf(F,df_time,df_error)
	else:
		P = None
	if output_eta_sq:
		partial_eta_sq = SStime / (SStime + SSerror)
	else:
		partial_eta_sq = None
	return(F, P, partial_eta_sq)


def rm_anova_one_bs_factor(data, between_factor, output_sig = False):
	"""
	One factor repeated measure ANOVA for longitudinal dependent variables
	
	Parameters
	----------
	data : array
		Data array (N_intervals, N_individuals, N_dependent variables)
	between_factor : array
		1D array of the between subject factor. 
	
	Optional Flags
	----------
	output_sig : bool
		outputs p-values of F-statistics
	
	Returns
	-------
	Fbetween : array
		F-statistics of the between subject factor
	Ftime : array
		F-statistics of the within-subject interval
	Fint : array
		F-statistics of the factor*interval interaction
	
	Optional returns
	-------
	Pbetween : array
		P-statistics of the between subject factor
	Ptime : array
		P-statistics of the within-subject interval
	Pint : array
		P-statistics of the factor*interval interaction
	
	"""

	k = data.shape[0]
	Ns =  data.shape[1]

	#grand mean
	mu_grand = 0
	for i in range(k):
		mu_grand += np.mean(data[i],0)
	mu_grand = np.divide(mu_grand,k)

	# Total sum of squares
	if data.ndim == 3:
		SStotal = np.sum((data - mu_grand)**2, (0,1))
	else:
		SStotal = np.sum((data - mu_grand)**2)


	# Sum of squareas of the subjects
	SSsub = np.sum((np.mean(data,0) - mu_grand)**2,0) * k

	ni_array = []
	for factor in np.unique(between_factor):
		ni = data[:,between_factor==factor].shape[1]
		ni_array.append(ni)
	ni = np.divide(np.sum(np.array(ni_array)), k)
#	print(ni)

	# Sum of squares of the groups
	SSgroups = 0
	for factor in np.unique(between_factor):
		if data.ndim == 3:
			SSgroups += (np.mean(data[:,between_factor==factor], (0,1)) - mu_grand)**2 * k * data[:,between_factor==factor].shape[1]
		else:
			SSgroups += (np.mean(data[:,between_factor==factor]) - mu_grand)**2 * k * data[:,between_factor==factor].shape[1]

	# Sum of squares intervals (i.e., time)
	ki = len(np.unique(between_factor))
	SStime = np.sum((np.mean(data,1) - mu_grand)**2,0) * ni * ki

	# Sum of squares for cells
	SScells = 0

	for factor in np.unique(between_factor):
		SScells += np.sum(np.square(np.mean(data[:,between_factor==factor],1) - mu_grand),0) * data[:,between_factor==factor].shape[1]

	SSint = SScells - SStime - SSgroups
	SSwithingroups = SSsub - SSgroups

	SSwithinsubs = SStotal - SSsub
	SSintwithingroups = SSwithinsubs - SStime - SSint

	df_total = k * Ns - 1
	df_groups = ki - 1
	df_withingroups = (ki * (ni - 1))
	df_time = (k-1)
	df_int = df_groups * df_time
	df_intwithingroups = df_total - df_groups - df_withingroups - df_time - df_int

	MSgroups = np.divide(SSgroups, df_groups)
	MSwithingroups = np.divide(SSwithingroups, df_withingroups)
	Fbetween = np.divide(MSgroups, MSwithingroups)


	MStime = np.divide(SStime, df_time)
	MSint = np.divide(SSint, df_int)
	MSintwithingroups = np.divide(SSintwithingroups, df_intwithingroups)

	Ftime =  np.divide(MStime, MSintwithingroups)
	Fint =  np.divide(MSint, MSintwithingroups)

	if output_sig:
		Pbetween = 1 - f.cdf(Fbetween,df_groups,df_withingroups)
		Ptime = 1 - f.cdf(Ftime,df_time,df_intwithingroups)
		Pint = 1 - f.cdf(Fint,df_int,df_intwithingroups)
		return(Fbetween, Ftime, Fint, Pbetween, Ptime, Pint)
	else:
		return(Fbetween, Ftime, Fint)


def full_glm_results(endog_arr, exog_vars, only_tvals = False, return_resids = False, return_fitted = False, PCA_whiten = False, ZCA_whiten = False,  orthogonalize = False, orthogNear = False, orthog_GramSchmidt = False):
	"""
	One factor repeated measure ANOVA for longitudinal dependent variables
	
	Parameters
	----------
	endog_arr : array
		endogenours variable array (i.e., masked neuroimage dependent variables)
	exog_vars : array
		exogenous variable array (i.e., independent variables)
	
	Optional Flags
	----------

	only_tvals : bool
		Returns only t-values (default = False)
	return_resids : bool
		Returns the residuals (default = False)
	return_fitted : bool
		Returns the fitted values (default = False)
	PCA_whiten : bool
		PCA whiten indendent variables (not recommended) (default = False)
	PCA_whiten : bool 
		ZCA whiten indendent variables (not recommended) (default = False)
	PCA_whiten : bool 
		ZCA whiten indendent variables (not recommended) (default = False)
	orthogonalize : bool 
		Orthogonalize exogenous variables (default = True)
	orthogNear : bool 
		Orthogonalize exogenous variables to the nearest orthogonal matrix (default = False)
	orthog_GramSchmidt : bool
		Use the Gram-Schmidt process to orthonormalize the exogenous variables.
		i.e., Type-1 sum of squares (like R).
		It is particularly useful for assessing interations.
	
	Returns
	-------
	Fvalues  :array
		F-statistics of model
	Tvalues : array
		T-statistics of factors
	Pvalues : array
		P-values of factors
	R2 : array
		R-squared of the model
	R2_adj : array
		Adjusted R-squared of the model
	
	Optional returns
	-------
	resids : array
		The residuals of the model
	fitted : array
		The fitted values of the model
	
	"""
	if np.mean(exog_vars[:,0])!=1:
		print("Warning: the intercept is not included as the first column in your exogenous variable array")
	n, num_depv = endog_arr.shape
	k = exog_vars.shape[1]

	if orthogonalize:
		exog_vars = stack_ones(orthog_columns(exog_vars[:,1:]))
	if orthogNear:
		exog_vars = stack_ones(ortho_neareast(exog_vars[:,1:]))
	if orthog_GramSchmidt: # for when order matters AKA type 2 sum of squares
		exog_vars = stack_ones(gram_schmidt_orthonorm(exog_vars[:,1:]))

	invXX = np.linalg.inv(np.dot(exog_vars.T, exog_vars))

	DFbetween = k - 1 # aka df model
	DFwithin = n - k # aka df residuals
	DFtotal = n - 1
	if PCA_whiten:
		endog_arr = PCAwhiten(endog_arr)
	if ZCA_whiten:
		endog_arr = ZCAwhiten(endog_arr)

	a = cy_lin_lstsqr_mat(exog_vars, endog_arr)
	sigma2 = np.sum((endog_arr - np.dot(exog_vars,a))**2,axis=0) / (n - k)
	se = se_of_slope(num_depv,invXX,sigma2,k)

	if only_tvals:
		return a / se
	else:
		resids = endog_arr - np.dot(exog_vars,a)
		RSS = np.sum(resids**2,axis=0)
		TSS = np.sum((endog_arr - np.mean(endog_arr, axis =0))**2, axis = 0)
		R2 = 1 - (RSS/TSS)
		R2_adj = 1 - ((1-R2)*DFtotal/(DFwithin))
		Fvalues = ((TSS-RSS)/(DFbetween))/(RSS/DFwithin)
		Tvalues = a / se
		Pvalues = t.sf(np.abs(Tvalues), DFtotal)*2
		if return_resids:
			if return_fitted:
				fitted = np.dot(exog_vars, a)
				return (Tvalues, Pvalues, Fvalues, R2, R2_adj, np.array(resids), np.array(fitted))
			else:
				return (Tvalues, Pvalues, Fvalues, R2, R2_adj, np.array(resids))
		else:
			return (Tvalues, Pvalues, Fvalues, R2, R2_adj)

# Slow but safer
def rm_anova_two_bs_factor(data, factor1, factor2, output_sig = False):
	"""
	Two factor repeated measure ANOVA for longitudinal dependent variables
	
	Parameters
	----------
	data : array
		Data array (N_intervals, N_individuals, N_dependent variables)
	factor1 : array
		1D array of factor 1
	factor2 : array
		1D array of factor 2
	
	Optional Flags
	----------
	output_sig : bool
		outputs p-values of F-statistics
	
	Returns
	-------
	F_a : array
		F-statistics of the between-subject factor1
	F_b : array
		F-statistics of the between-subject factor1
	F_ab : array
		F-statistics of the between-subject interaction of factor1*factor2
	F_s : array
		F-statistics of the within-subject interval
	F_sb : array
		F-statistics of the factor1*interval interaction
	F_sb : array
		F-statistics of the factor2*interval interaction
	F_sab : array
		F-statistics of the factor1*factor2*interval interaction
	
	Optional returns
	-------
	
	P_a : array
		P-statistics of the between-subject factor1
	P_b : array
		P-statistics of the between-subject factor1
	P_ab : array
		P-statistics of the between-subject interaction of factor1*factor2
	P_s : array
		P-statistics of the within-subject interval
	P_sa : array
		P-statistics of the factor1*interval interaction
	P_sb : array
		P-statistics of the factor2*interval interaction
	P_sab : array
		P-statistics of the factor1*factor2*interval interaction
	
	"""
	if data.ndim == 2:
		data = data[:,:,np.newaxis]

	# merge the factors for the interaction
	f1xf2 = np.array(map('_'.join, zip(factor1.astype(np.str),factor2.astype(np.str))))

	s = data.shape[0]
	n = data.shape[1]
	a = len(np.unique(factor1)) 
	b = len(np.unique(factor2))

	#grand mean
	mu_grand = np.mean(data,(0,1))

	# Total sum of squares
	SS_Total = np.sum((data - mu_grand)**2, (0,1))
	SS_BetweenSubjects = np.sum((np.mean(data,0) - mu_grand)**2,0) * s
	SS_WithinSubjects = SS_Total - SS_BetweenSubjects

	# decomposition matrix
	DecompMat = np.empty((s*n,7,data.shape[2]))

	data_long = data.reshape(s*n,data.shape[2])
	factor1_long = factor1
	factor2_long = factor2
	f1xf2_long = f1xf2
	for i in range(s-1):
		factor1_long = np.append(factor1_long,factor1)
		factor2_long = np.append(factor2_long,factor2)
		f1xf2_long = np.append(f1xf2_long,f1xf2)

	interval_long = np.zeros_like(factor1_long)
	for i in range(s):
		interval_long[(i*n):(i*n+n)] = np.ones((n)) * i

	# BETWEEN SUBJECT EFFECTS
	#a
	for factor in np.unique(factor1):
		DecompMat[factor1_long == factor,0,:] = np.mean(data_long[factor1_long==factor,:] - mu_grand,0)
	#b 
	for factor in np.unique(factor2):
		DecompMat[factor2_long == factor,1,:] = np.mean(data_long[factor2_long==factor,:] - mu_grand,0)
	for factor in np.unique(f1xf2_long):
		DecompMat[f1xf2_long == factor,2,:] = np.mean(data_long[f1xf2_long==factor,:] - mu_grand,0)
	DecompMat[:,2,:] = DecompMat[:,2,:] - DecompMat[:,0,:] - DecompMat[:,1,:]

	SS_a = np.sum(np.square(DecompMat[:,0,:]),0)
	SS_b = np.sum(np.square(DecompMat[:,1,:]),0)
	SS_ab = np.sum(np.square(DecompMat[:,2,:]),0)
	SS_WithinFactors = SS_BetweenSubjects - SS_a - SS_b - SS_ab


	# WITHIN SUBJECT EFFECTS
	for i in range(s):
		DecompMat[(i*n):(i*n+n),3,:] = data[i].mean(0) - mu_grand

	for i in range(s):
		for factor in np.unique(factor1_long):
			DecompMat[(i*n):(i*n+n),4,:][factor1 == factor] = np.mean(data_long[(interval_long == i)*(factor1_long == factor),:],0) - mu_grand
	DecompMat[:,4,:] = DecompMat[:,4,:] - DecompMat[:,0,:] - DecompMat[:,3,:]

	for i in range(s):
		for factor in np.unique(factor2_long):
			DecompMat[(i*n):(i*n+n),5,:][factor2 == factor] = np.mean(data_long[(interval_long == i)*(factor2_long == factor),:],0) - mu_grand
	DecompMat[:,5,:] = DecompMat[:,5,:] - DecompMat[:,1,:] - DecompMat[:,3,:]

	for i in range(s):
		for factor in np.unique(f1xf2_long):
			DecompMat[(i*n):(i*n+n),6,:][f1xf2 == factor] = np.mean(data_long[(interval_long == i)*(f1xf2_long == factor),:],0) - mu_grand
	DecompMat[:,6,:] = DecompMat[:,6,:] - DecompMat[:,0,:] -DecompMat[:,1,:] - DecompMat[:,2,:] - DecompMat[:,3,:] - DecompMat[:,4,:] - DecompMat[:,5,:]

	SS_s = np.sum(np.square(DecompMat[:,3,:]),0)
	SS_sa = np.sum(np.square(DecompMat[:,4,:]),0)
	SS_sb = np.sum(np.square(DecompMat[:,5,:]),0)
	SS_sab = np.sum(np.square(DecompMat[:,6,:]),0)
	SS_sWithinFactors = SS_WithinSubjects - SS_s - SS_sa - SS_sb - SS_sab

	# Between subjects df
	df_BetweenSubjects  = n - 1
	df_a =  a - 1
	df_b =  b - 1
	df_ab =  df_a * df_b
	df_WithinFactors = df_BetweenSubjects - df_a - df_b - df_ab

	# Within subjects df
	df_s = s - 1
	df_sa = df_a * df_s
	df_sb = df_b * df_s
	df_sab = df_ab * df_s
	df_sWithinFactor = df_WithinFactors * df_s

	# F-stats
	# Between subjects
	ms_WithinFactors = np.divide(SS_WithinFactors, df_WithinFactors)
	F_a = np.divide(np.divide(SS_a, df_a), ms_WithinFactors)
	F_b = np.divide(np.divide(SS_b, df_b), ms_WithinFactors)
	F_ab = np.divide(np.divide(SS_ab, df_ab), ms_WithinFactors)

	# Within subjects
	ms_sWithinFactor = np.divide(SS_sWithinFactors, df_sWithinFactor)
	F_s = np.divide(np.divide(SS_s, df_s), ms_sWithinFactor)
	F_sa = np.divide(np.divide(SS_sa, df_sa), ms_sWithinFactor)
	F_sb = np.divide(np.divide(SS_sb, df_sb), ms_sWithinFactor)
	F_sab = np.divide(np.divide(SS_sab, df_sab), ms_sWithinFactor)

	if output_sig:
		# Between subjects
		P_a = 1 - f.cdf(F_a,df_a,df_WithinFactors)
		P_b = 1 - f.cdf(F_b,df_b,df_WithinFactors)
		P_ab = 1 - f.cdf(F_ab,df_ab,df_WithinFactors)
		# Within subjects
		P_s = 1 - f.cdf(F_s,df_s,df_sWithinFactor)
		P_sa = 1 - f.cdf(F_sa,df_sa,df_sWithinFactor)
		P_sb = 1 - f.cdf(F_sb,df_sb,df_sWithinFactor)
		P_sab = 1 - f.cdf(F_sab,df_sab,df_sWithinFactor)
		return (F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab, P_a, P_b, P_ab, P_s, P_sa, P_sb, P_sab)
	else:
		return (F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab)

# convert to require dummy coded variables
# dmy_factor1 = dummy_code(pdData.sexM1)
# dmy_factor2 = dummy_code(pdData.S4_PRS_GF_noIMAGEN,iscontinous=True, demean=True)
# dmy_subjects = dummy_code(pdData.SubjID, demean=False)
# dmy_covariates = dummy_code(pdData.site, demean=True, iscontinous=False)
# Check if QR orthog is faster... 
def reg_rm_ancova_two_bs_factor(data, dmy_factor1, dmy_factor2, dmy_subjects, dmy_covariates = None, data_format = 'short', output_sig = False, verbose = True, rand_array = None, use_reduced_residuals = False, output_reduced_residuals = False):
	"""
	Two factor repeated measure ANCOVA for longitudinal dependent variables
	Note: Type 1 Sum of Squares is used, therefore order matters
	
	Parameters
	----------
	data : array
		Data array (N_intervals, N_individuals, N_dependent variables)
	dmy_factor1 : array
		dummy coded factor 1
	dmy_factor2 : array
		dummy coded factor 2
	dmy_subjects : array
		dummy coded subjects
	
	Optional Parameters
	----------
	dmy_covariates : array
		dummy coded covariates of no interest
	rand_array : arrays
		randomised shuffled array (used for permutation testing)
	
	Optional Flags
	----------
	output_sig : bool
		outputs p-values of F-statistics
	verbose : bool
		prints the max F-statistics values and degrees of freedom
	use_reduced_residuals : bool
		The residuals after regressing the fixed effect exogenous variable(s) is shuffled for permutations testing.
	output_reduced_residuals : bool
		Outputs the residuals array after regressing the fixed effect exogenous variable(s).
	
	Returns
	-------
	F_a : array
		F-statistics of the between-subject factor1
	F_b : array
		F-statistics of the between-subject factor2
	F_ab : array
		F-statistics of the between-subject interaction of factor1*factor2
	F_s : array
		F-statistics of the within-subject interval
	F_sb : array
		F-statistics of the factor1*interval interaction
	F_sb : array
		F-statistics of the factor2*interval interaction
	F_sab : array
		F-statistics of the factor1*factor2*interval interaction
	
	Optional returns
	-------
	
	P_a : array
		P-statistics of the between-subject factor1
	P_b : array
		P-statistics of the between-subject factor2
	P_ab : array
		P-statistics of the between-subject interaction of factor1*factor2
	P_s : array
		P-statistics of the within-subject interval
	P_sa : array
		P-statistics of the factor1*interval interaction
	P_sb : array
		P-statistics of the factor2*interval interaction
	P_sab : array
		P-statistics of the factor1*factor2*interval interaction
	
	OR
	
	reduced_data : array
		Residuals array after regressing the fixed effect exogenous variable(s) in long format.
		
	"""
	
	if data_format == 'short':
		if data.ndim == 2:
			data = data[:,:,np.newaxis]
		# get shapes for df
		n = data.shape[1]
		s = data.shape[0]
		endog_arr = data.reshape(s*n,data.shape[2])
	elif data_format == 'long':
		n = len(dmy_factor1)
		s = len(data)/n
		endog_arr = data
	else:
		print("Error: data format must be short or long.")
	del data # reduce ram usage
	
	if dmy_factor1.ndim == 1:
		fa = 2
	else:
		fa = dmy_factor1.shape[1] + 1
	if dmy_factor2.ndim == 1:
		fb = 2
	else:
		fb = dmy_factor2.shape[1] + 1

	if rand_array is not None:
		if use_reduced_residuals:
			r_dmy_factor1_long = dmy_factor1
			r_dmy_factor2_long = dmy_factor2
			r_dmy_interaction_long = column_product(dmy_factor1,dmy_factor2)
			if dmy_covariates is not None:
				r_dmy_covars_long = dmy_covariates
			for i in range(s-1):
				r_dmy_factor1_long = np.concatenate((r_dmy_factor1_long,dmy_factor1),0)
				r_dmy_factor2_long = np.concatenate((r_dmy_factor2_long,dmy_factor2),0)
				r_dmy_interaction_long = np.concatenate((r_dmy_interaction_long,column_product(dmy_factor1,dmy_factor2)),0)
				if dmy_covariates is not None:
					r_dmy_covars_long = np.concatenate((r_dmy_covars_long, dmy_covariates),0)
			exog_vars = stack_ones(r_dmy_factor1_long)
			exog_vars = np.column_stack((exog_vars, r_dmy_factor2_long))
			exog_vars = np.column_stack((exog_vars, r_dmy_interaction_long))
			if dmy_covariates is not None:
				exog_vars = np.column_stack((exog_vars, r_dmy_covars_long))
			a = cy_lin_lstsqr_mat(exog_vars,endog_arr)
			endog_arr = endog_arr - np.dot(exog_vars,a)
			del a
		np.random.shuffle(endog_arr)
		dmy_factor1 = dmy_factor1[rand_array]
		dmy_factor2 = dmy_factor2[rand_array]
		if dmy_covariates is not None:
			dmy_covariates = dmy_covariates[rand_array]

	# code the interaction
	dmy_interaction = column_product(dmy_factor1, dmy_factor2)

	# convert to long form
	interval_long = np.zeros(n)
	dmy_factor1_long = dmy_factor1
	dmy_factor2_long = dmy_factor2
	dmy_interaction_long = dmy_interaction
	dmy_subjects_long = dmy_subjects
	if dmy_covariates is not None:
		dmy_covars_long = dmy_covariates

	for i in range(s-1):
		dmy_factor1_long = np.concatenate((dmy_factor1_long,dmy_factor1),0)
		dmy_factor2_long = np.concatenate((dmy_factor2_long,dmy_factor2),0)
		dmy_interaction_long = np.concatenate((dmy_interaction_long,dmy_interaction),0)
		dmy_subjects_long = np.concatenate((dmy_subjects_long,dmy_subjects),0)
		interval_long = np.concatenate((interval_long, np.ones(n)*(i+1)))
		if dmy_covariates is not None:
			dmy_covars_long = np.concatenate((dmy_covars_long, dmy_covariates),0)
	dmy_interval_long = dummy_code(interval_long, demean = False)

	# SS Totals
	exog_vars = stack_ones(dmy_subjects_long)
	SS_WithinSubjects = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_Total = np.sum((endog_arr - np.mean(endog_arr, axis = 0))**2, axis = 0)
	SS_BetweenSubjects = SS_Total - SS_WithinSubjects

	# SS between subject
	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_factor2_long))
	exog_vars = np.column_stack((exog_vars, dmy_interaction_long))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
	if output_reduced_residuals:
		a, residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)
		reduced_data = endog_arr - np.dot(exog_vars,a)
		del a
	else:
		residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_cells_ab = SS_Total - residuals

	exog_vars = stack_ones(np.column_stack((dmy_factor1_long, dmy_factor2_long)))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_ab = SS_cells_ab - (SS_Total - residuals)
	# correct SS_ab

	if dmy_covariates is not None:
		exog_vars = stack_ones(dmy_covars_long)
		residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
		SS_covars =(SS_Total - residuals)
	else:
		SS_covars = 0

	# SSb
	exog_vars = stack_ones(dmy_factor1_long)
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_a = (SS_Total - residuals) - SS_covars

	# SSa
	SS_b = SS_cells_ab - SS_covars - SS_ab - SS_a

	SS_WithinFactors = SS_BetweenSubjects - SS_a - SS_b - SS_ab - SS_covars

	# SS time
	exog_vars = stack_ones(dmy_interval_long)
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_s =  SS_Total - residuals

	# SS Cells s*a*b
	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_factor2_long))
	exog_vars = np.column_stack((exog_vars, dmy_interaction_long))
	exog_vars = np.column_stack((exog_vars, dmy_interval_long))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor1_long,dmy_interval_long)))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor2_long,dmy_interval_long)))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_interaction_long,dmy_interval_long)))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_cells_sab = SS_Total - residuals

	# SS Cells s*a*b
	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_factor2_long))
	exog_vars = np.column_stack((exog_vars, dmy_interaction_long))
	exog_vars = np.column_stack((exog_vars, dmy_interval_long))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor1_long,dmy_interval_long)))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor2_long,dmy_interval_long)))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_sab = SS_cells_sab - (SS_Total - residuals)

	# SS covariates (same as above int then minus int)
	if dmy_covariates is not None:
		exog_vars = stack_ones(dmy_covars_long)
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
		exog_vars = np.column_stack((exog_vars, dmy_interval_long))
		residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
		SS_cells_scovars = (SS_Total - residuals)
		exog_vars = stack_ones(dmy_covars_long)
		exog_vars = np.column_stack((exog_vars, dmy_interval_long))
		residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
		SS_scovars = SS_cells_scovars - (SS_Total - residuals)
	else:
		SS_scovars = 0

	# SS Cells s*a
	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_interval_long))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor1_long,dmy_interval_long)))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_Cells_sa = (SS_Total - residuals)

	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_interval_long))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_sa = SS_Cells_sa - (SS_Total - residuals)

	# SS Cells s*b
	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_factor2_long))
	exog_vars = np.column_stack((exog_vars, dmy_interval_long))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor1_long,dmy_interval_long)))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor2_long,dmy_interval_long)))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_Cells_sb = (SS_Total - residuals)

	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_factor2_long))
	exog_vars = np.column_stack((exog_vars, dmy_interval_long))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor1_long,dmy_interval_long)))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_sb = SS_Cells_sb - (SS_Total - residuals)

	SS_sWithinFactors = SS_WithinSubjects - SS_s - SS_sa - SS_sb - SS_sab - SS_scovars

	# Between subjects df
	df_BetweenSubjects  = n - 1
	df_a =  fa - 1
	df_b =  fb - 1
	df_ab =  df_a * df_b
	if dmy_covariates is not None:
		if dmy_covariates.ndim == 1:
			df_covars = 1
		else:
			df_covars = dmy_covariates.shape[1]
	else:
		df_covars = 0
	df_WithinFactors = df_BetweenSubjects - df_a - df_b - df_ab - df_covars

	# Within subjects df
	df_s = s - 1
	df_sa = df_a * df_s
	df_sb = df_b * df_s
	df_sab = df_ab * df_s
	df_sWithinFactor = df_WithinFactors * df_s

	# F-stats
	# Between subjects
	ms_WithinFactors = np.divide(SS_WithinFactors, df_WithinFactors)
	F_a = np.divide(np.divide(SS_a, df_a), ms_WithinFactors)
	F_b = np.divide(np.divide(SS_b, df_b), ms_WithinFactors)
	F_ab = np.divide(np.divide(SS_ab, df_ab), ms_WithinFactors)

	# Within subjects
	ms_sWithinFactor = np.divide(SS_sWithinFactors, df_sWithinFactor)
	F_s = np.divide(np.divide(SS_s, df_s), ms_sWithinFactor)
	F_sa = np.divide(np.divide(SS_sa, df_sa), ms_sWithinFactor)
	F_sb = np.divide(np.divide(SS_sb, df_sb), ms_sWithinFactor)
	F_sab = np.divide(np.divide(SS_sab, df_sab), ms_sWithinFactor)

	if verbose:
		print("Source\t\tDF\tF(Max)")
		print("Factor1\t\t(%d,%d)\t%.2f" % (df_a, df_WithinFactors, F_a.max()))
		print("Factor2\t\t(%d,%d)\t%.2f" % (df_b, df_WithinFactors, F_b.max()))
		print("F1*F2\t\t(%d,%d)\t%.2f" % (df_ab, df_WithinFactors, F_ab.max()))
		print("Time\t\t(%d,%d)\t%.2f" % (df_s, df_sWithinFactor, F_s.max()))
		print("F1*Time\t\t(%d,%d)\t%.2f" % (df_sa, df_sWithinFactor, F_sa.max()))
		print("F2*Time\t\t(%d,%d)\t%.2f" % (df_sb, df_sWithinFactor, F_sb.max()))
		print("F1*F2*Time\t(%d,%d)\t%.2f" % (df_sab, df_sWithinFactor, F_sab.max()))

	if output_sig:
		# Between subjects
		P_a = 1 - f.cdf(F_a,df_a,df_WithinFactors)
		P_b = 1 - f.cdf(F_b,df_b,df_WithinFactors)
		P_ab = 1 - f.cdf(F_ab,df_ab,df_WithinFactors)
		# Within subjects
		P_s = 1 - f.cdf(F_s,df_s,df_sWithinFactor)
		P_sa = 1 - f.cdf(F_sa,df_sa,df_sWithinFactor)
		P_sb = 1 - f.cdf(F_sb,df_sb,df_sWithinFactor)
		P_sab = 1 - f.cdf(F_sab,df_sab,df_sWithinFactor)
		return (F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab, P_a, P_b, P_ab, P_s, P_sa, P_sb, P_sab)
	else:
		if output_reduced_residuals:
			return (F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab, reduced_data)
		else:
			return (F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab)



# convert to require dummy coded variables
# dmy_factor1 = dummy_code(pdData.sexM1)
# dmy_factor2 = dummy_code(pdData.S4_PRS_GF_noIMAGEN,iscontinous=True, demean=True)
# dmy_subjects = dummy_code(pdData.SubjID, demean=False)
# dmy_covariates = dummy_code(pdData.site, demean=True, iscontinous=False)
# Check if QR orthog is faster... 
#Kherad-Pajouh, S. & Renaud, O. Stat Papers (2015) 56: 947. https://doi.org/10.1007/s00362-014-0617-3

def reg_rm_ancova_one_bs_factor(data, dmy_factor1, dmy_subjects, data_format = 'short', dmy_covariates = None, output_sig = False, verbose = True, rand_array = None, use_reduced_residuals = False, output_reduced_residuals = False):
	"""
	One factor repeated measure ANCOVA for longitudinal dependent variables
	
	Parameters
	----------
	data : array
		Data array (N_intervals, N_individuals, N_dependent variables)
	dmy_factor1 : array
		dummy coded factor 1
	dmy_subjects : array
		dummy coded subjects
	
	Optional Parameters
	----------
	dmy_covariates : array
		dummy coded covariates of no interest
	
	Optional Flags
	----------
	output_sig : bool
		outputs p-values of F-statistics
	
	Returns
	-------
	F_a : array
		F-statistics of the between-subject factor1
	F_s : array
		F-statistics of the within-subject interval
	F_sa : array
		F-statistics of the factor1*interval interaction
	
	Optional returns
	-------
	
	P_a : array
		P-statistics of the between-subject factor1
	P_s : array
		P-statistics of the within-subject interval
	P_sa : array
		P-statistics of the factor1*interval interaction
	
	"""

	if data_format == 'short':
		if data.ndim == 2:
			data = data[:,:,np.newaxis]
		# get shapes for df
		s = data.shape[0]
		n = data.shape[1]
		endog_arr = data.reshape(s*n,data.shape[2])
	elif data_format == 'long':
		n = len(dmy_factor1)
		s = len(data)/n
		endog_arr = data
	else:
		print("Error: data format must be short or long.")

	#ram clean-up
	del data 

	if dmy_factor1.ndim == 1:
		fa = 2
	else:
		fa = dmy_factor1.shape[1] + 1

	if rand_array is not None:
		if use_reduced_residuals:
			r_dmy_factor1_long = dmy_factor1
			if dmy_covariates is not None:
				r_dmy_covars_long = dmy_covariates
			for i in range(s-1):
				r_dmy_factor1_long = np.concatenate((r_dmy_factor1_long,dmy_factor1),0)
				if dmy_covariates is not None:
					r_dmy_covars_long = np.concatenate((r_dmy_covars_long, dmy_covariates),0)
			exog_vars = stack_ones(r_dmy_factor1_long)
			if dmy_covariates is not None:
				exog_vars = np.column_stack((exog_vars, r_dmy_covars_long))
			a = cy_lin_lstsqr_mat(exog_vars,endog_arr)
			endog_arr = endog_arr - np.dot(exog_vars,a)
		np.random.shuffle(endog_arr)
		dmy_factor1 = dmy_factor1[rand_array]
		if dmy_covariates is not None:
			dmy_covariates = dmy_covariates[rand_array]

	# convert to exogenous variables to long form
	interval_long = np.zeros(n)
	dmy_factor1_long = dmy_factor1
	dmy_subjects_long = dmy_subjects
	if dmy_covariates is not None:
		dmy_covars_long = dmy_covariates
	for i in range(s-1):
		dmy_factor1_long = np.concatenate((dmy_factor1_long,dmy_factor1),0)
		dmy_subjects_long = np.concatenate((dmy_subjects_long,dmy_subjects),0)
		interval_long = np.concatenate((interval_long, np.ones(n)*(i+1)))
		if dmy_covariates is not None:
			dmy_covars_long = np.concatenate((dmy_covars_long, dmy_covariates),0)
	dmy_interval_long = dummy_code(interval_long, demean = False)

	# SS Totals
	exog_vars = stack_ones(dmy_subjects_long)
	endog_arr = np.squeeze(endog_arr)
	SS_WithinSubjects = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_Total = np.sum((endog_arr - np.mean(endog_arr, axis = 0))**2, axis = 0)
	SS_BetweenSubjects = SS_Total - SS_WithinSubjects

	# SS between subject
	exog_vars = stack_ones(dmy_factor1_long)
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
	if output_reduced_residuals:
		a, residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)
		reduced_data = endog_arr - np.dot(exog_vars,a)
		del a
	else:
		residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_cells_a = SS_Total - residuals

	if dmy_covariates is not None:
		exog_vars = stack_ones(dmy_covars_long)
		residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
		SS_covars =(SS_Total - residuals)
	else:
		SS_covars = 0

	SS_a = SS_cells_a - SS_covars
	SS_WithinFactors = SS_BetweenSubjects - SS_a - SS_covars

	# SS time
	exog_vars = stack_ones(dmy_interval_long)
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_s =  SS_Total - residuals

	# SS covariates (same as above int then minus int)
	if dmy_covariates is not None:
		exog_vars = stack_ones(dmy_covars_long)
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
		exog_vars = np.column_stack((exog_vars, dmy_interval_long))
		residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
		SS_cells_scovars = (SS_Total - residuals)
		exog_vars = stack_ones(dmy_covars_long)
		exog_vars = np.column_stack((exog_vars, dmy_interval_long))
		residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
		SS_scovars = SS_cells_scovars - (SS_Total - residuals)
	else:
		SS_scovars = 0

	# SS Cells s*a
	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_interval_long))
	exog_vars = np.column_stack((exog_vars, column_product(dmy_factor1_long,dmy_interval_long)))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_Cells_sa = (SS_Total - residuals)

	exog_vars = stack_ones(dmy_factor1_long)
	exog_vars = np.column_stack((exog_vars, dmy_interval_long))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covars_long))
		exog_vars = np.column_stack((exog_vars, column_product(dmy_covars_long,dmy_interval_long)))
	residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog_arr)[1]
	SS_sa = SS_Cells_sa - (SS_Total - residuals)

	SS_sWithinFactors = SS_WithinSubjects - SS_s - SS_sa - SS_scovars

	# Between subjects df
	df_BetweenSubjects  = n - 1
	df_a =  fa - 1
	if dmy_covariates is not None:
		if dmy_covariates.ndim == 1:
			df_covars = 1
		else:
			df_covars = dmy_covariates.shape[1]
	else:
		df_covars = 0
	df_WithinFactors = df_BetweenSubjects - df_a - df_covars

	# Within subjects df
	df_s = s - 1
	df_sa = df_a * df_s
	df_sWithinFactor = df_WithinFactors * df_s

	# F-stats
	# Between subjects
	ms_WithinFactors = np.divide(SS_WithinFactors, df_WithinFactors)
	F_a = np.divide(np.divide(SS_a, df_a), ms_WithinFactors)

	# Within subjects
	ms_sWithinFactor = np.divide(SS_sWithinFactors, df_sWithinFactor)
	F_s = np.divide(np.divide(SS_s, df_s), ms_sWithinFactor)

	F_sa = np.divide(np.divide(SS_sa, df_sa), ms_sWithinFactor)

	if verbose:
		print("Source\t\tDF\tF(Max)")
		print("Factor\t\t(%d,%d)\t%.2f" % (df_a, df_WithinFactors, F_a.max()))
		print("Time\t\t(%d,%d)\t%.2f" % (df_s, df_sWithinFactor, F_s.max()))
		print("Factor*Time\t(%d,%d)\t%.2f" % (df_sa, df_sWithinFactor, F_sa.max()))

	if output_sig:
		# Between subjects
		P_a = 1 - f.cdf(F_a,df_a,df_WithinFactors)
		# Within subjects
		P_s = 1 - f.cdf(F_s,df_s,df_sWithinFactor)
		P_sa = 1 - f.cdf(F_sa,df_sa,df_sWithinFactor)
		return (F_a, F_s, F_sa, P_a, P_s, P_sa)
	else:
		if output_reduced_residuals:
			return (F_a, F_s, F_sa, reduced_data)
		else:
			return (F_a, F_s, F_sa)

# Type I Sum of Squares (order matters!!!)
def glm_typeI(endog, exog, dmy_covariates = None, output_fvalues = True, output_tvalues = False, output_pvalues = False, verbose = True, rand_array = None, use_reduced_residuals = False, output_reduced_residuals = False, exog_names = None):
	"""
	Generalized ANCOVA using Type I Sum of Squares
	
	Parameters
	----------
	endog : array
		Endogenous (dependent) variable array (Nsubjects, Nvariables)
	exog : array
		Exogenous (independent) dummy coded variables
		exog is an array of arrays (Nvariables, Nsubjects, Kvariable)
	dmy_covariates : array
		Dummy coded array of covariates of no interest
	
	Returns
	---------
	To-do
	"""
	
	n = endog.shape[0]
	
	kvars = []
	exog_vars = np.ones((n))
	for var in exog:
		var = np.array(var)
		if var.ndim == 1:
			kvars.append((1))
		else:
			kvars.append((var.shape[1]))
		exog_vars = np.column_stack((exog_vars,var))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covariates))
	exog_vars = np.array(exog_vars)

	if rand_array is not None:
		if use_reduced_residuals:
			a = cy_lin_lstsqr_mat(exog_vars,endog)
			endog = endog - np.dot(exog_vars,a)
		exog_vars = exog_vars[rand_array]

	if output_reduced_residuals:
		a = cy_lin_lstsqr_mat(exog_vars,endog)
		reduced_data = endog - np.dot(exog_vars,a)

	k = exog_vars.shape[1]

	DF_Between = k - 1 # aka df model
	DF_Within = n - k # aka df residuals
	DF_Total = n - 1

	SS_Total = np.sum((endog - np.mean(endog,0))**2,0)
	a, SS_Residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog)

	if output_fvalues:
		SS_Between = SS_Total - SS_Residuals
		MS_Residuals = (SS_Residuals/DF_Within)
		Fvalues = (SS_Between/DF_Between) / MS_Residuals

		if verbose:
			print("Source\t\tDF\tF(Max)")
			print("Model\t\t(%d,%d)\t%.2f" % (DF_Between, DF_Within, Fvalues.max()))

		# F value for exog
		Fvar = []
		Pvar = []
		start = 1
		for i, col in enumerate(kvars):
			stop = start + col
			SS_model = np.array(SS_Total - cy_lin_lstsqr_mat_residual(np.delete(exog_vars,np.s_[start:stop],1),endog)[1])
			Ftemp = (SS_Between - SS_model)/(MS_Residuals*kvars[i])
			Fvar.append(Ftemp)
			if verbose:
				if exog_names is not None:
					print("%s\t\t(%d,%d)\t%.2f" % (exog_names[i], col, DF_Within, Ftemp.max()))
				else:
					print("Exog%d\t\t(%d,%d)\t%.2f" % ((i+1), col, DF_Within, Ftemp.max()))
			start += col
			if output_pvalues:
				Pvar.append(f.sf(Fvar[i],col,DF_Within))

	if output_tvalues:
		sigma2 = np.sum((endog - np.dot(exog_vars,a))**2,axis=0) / (n - k)
		invXX = np.linalg.inv(np.dot(exog_vars.T, exog_vars))
		if endog.ndim == 1:
			se = np.sqrt(np.diag(sigma2 * invXX))
		else:
			num_depv = endog.shape[1]
			se = se_of_slope(num_depv,invXX,sigma2,k)
		Tvalues = a / se
	# return values
	if output_tvalues and output_fvalues:
		if output_pvalues:
			Pmodel = f.sf(Fvalues,DF_Between,DF_Within)
			Pvalues = t.sf(np.abs(Tvalues), DF_Total)*2
			return (Fvalues, np.array(Fvar), Tvalues, Pmodel, np.array(Pvar), Pvalues)
		else:
			if output_reduced_residuals:
				return (Fvalues, np.array(Fvar), Tvalues, reduced_data)
			else:
				return (Fvalues, np.array(Fvar), Tvalues)
	elif output_tvalues:
		if output_pvalues:
			Pvalues = t.sf(np.abs(Tvalues), DF_Total)*2
			return (Tvalues, Pvalues)
		else:
			if output_reduced_residuals:
				return (Tvalues, reduced_data)
			else:
				return Tvalues
	elif output_fvalues:
		if output_pvalues:
			Pmodel = f.sf(Fvalues,DF_Between,DF_Within)
			return (Fvalues, np.array(Fvar), Pmodel, np.array(Pvar))
		else:
			if output_reduced_residuals:
				return (Fvalues, np.array(Fvar), reduced_data)
			else:
				return (Fvalues, np.array(Fvar))
	else:
		print("No output has been selected")


# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3991883/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3663600/
def glm_cosinor(endog, time_var, exog = None, dmy_covariates = None, rand_array = None, interaction_var = None, period = [24.0], calc_MESOR = True, output_fit_only = False):
	"""
	COSINOR model using GLM
	
	Parameters
	----------
	endog : array
		Endogenous (dependent) variable array (Nsubjects, Nvariables)
	time_var : array
		Time variable [0-23.99] (Nsubjects).
	exog : array
		Exogenous (independent) dummy coded variables
		exog is an array of arrays (Nvariables, Nsubjects, Kvariable).
	dmy_covariates : array
		Dummy coded array of covariates of no interest.
	init_covars : array
		Dummy coded array of covariates for two-step regression.
	rand_array : array
		randomized array for permutations (Nsubjects).
	period : array
		Period(s) as an array of floats for cosinor model.
	Returns
	---------
	To-do
	"""

	n = endog.shape[0]
	# add cosinor terms
	num_period = len(period)
	exog_vars = np.ones((n))
	for i in range(num_period):
		exog_vars = np.column_stack((exog_vars,np.cos(np.divide(2.0*np.pi*time_var, period[i]))))
		exog_vars = np.column_stack((exog_vars,np.sin(np.divide(2.0*np.pi*time_var, period[i]))))

	if interaction_var is not None:
		for i in range(num_period):
			exog_vars = np.column_stack((exog_vars, exog_vars[i+1] * interaction_var))


	kvars = []
	# add other exogenous variables to the model (currently not implemented)
	if exog is not None:
		for var in exog:
			var = np.array(var)
			if var.ndim == 1:
				kvars.append((3))
			else:
				kvars.append((var.shape[1]))
			exog_vars = np.column_stack((exog_vars,var))

	# add covariates (i.e., exogenous variables that will not be outputed)
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covariates))
	exog_vars = np.array(exog_vars)

	if rand_array is not None:
		exog_vars = exog_vars[rand_array]

	# calculate model fit (Fmodel and R-sqr)
	k = exog_vars.shape[1]
	DF_Between = k - 1 # aka df model
	DF_Within = n - k # aka df residuals
	#DF_Total = n - 1

	a, SS_Residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog)
	if output_fit_only:
		AMPLITUDE = []
		ACROPHASE = []
		MESOR = a[0]
		for i in range(num_period):
			# beta, gamma
			AMPLITUDE.append(np.sqrt((a[1+(i*2),:]**2) + (a[2+(i*2),:]**2)))
			# Acrophase calculation
			if i == 0: # awful hack
				ACROPHASE = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				ACROPHASE = ACROPHASE[np.newaxis,:]
			else:
				temp_acro = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				temp_acro = temp_acro[np.newaxis,:]
				ACROPHASE = np.append(ACROPHASE,temp_acro, axis=0)
			ACROPHASE = np.array(ACROPHASE)
			ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)] = -ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)]
			ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)] = (-1*np.pi) + ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)]
			ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)] = (-1*np.pi) - ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)]
			ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)] = (-2*np.pi) + ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)]
		return MESOR, np.array(AMPLITUDE), np.array(ACROPHASE)
	else:
		SS_Total = np.sum((endog - np.mean(endog,0))**2,0)
		SS_Between = SS_Total - SS_Residuals
		MS_Residuals = (SS_Residuals / DF_Within)
		Fmodel = (SS_Between/DF_Between) / MS_Residuals
		# Calculates sigma sqr and T-value (intercept) for MESOR
		sigma = np.sqrt(SS_Residuals / DF_Within)
		invXX = np.linalg.inv(np.dot(exog_vars.T, exog_vars))

		if (calc_MESOR) or (exog is not None):
			if endog.ndim == 1:
				se = np.sqrt(np.diag(sigma * sigma * invXX))
				Tvalues = a / se
				MESOR = a[0]
				tMESOR = Tvalues[0]
				SE_MESOR = se[0]
				a = a[:, np.newaxis]
			else:
				num_depv = endog.shape[1]
				se = se_of_slope(num_depv,invXX,sigma**2,k)
				Tvalues = a / se
				MESOR = a[0,:]
				tMESOR = Tvalues[0,:]
				SE_MESOR = se[0,:]
			if exog is not None:
				tEXOG = Tvalues[(3+(2*(num_period-1))):,:]
			else:
				tEXOG = None
		else:
			MESOR = tMESOR = SE_MESOR = tEXOG = None

		AMPLITUDE = []
		ACROPHASE = []
		SE_ACROPHASE = []
		SE_AMPLITUDE = []
		tAMPLITUDE = []
		tACROPHASE = []
		
		for i in range(num_period):
			# beta, gamma
			AMPLITUDE.append(np.sqrt((a[1+(i*2),:]**2) + (a[2+(i*2),:]**2)))
			# Acrophase calculation
			if i == 0: # awful hack
				ACROPHASE = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				ACROPHASE = ACROPHASE[np.newaxis,:]
			else:
				temp_acro = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				temp_acro = temp_acro[np.newaxis,:]
				ACROPHASE = np.append(ACROPHASE,temp_acro, axis=0)

			# standard errors from error propagation
			SE_ACROPHASE.append(sigma * np.sqrt((invXX[(1+(i*2)),1+(i*2)]*np.sin(ACROPHASE[i])**2) + (2*invXX[1+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])*np.cos(ACROPHASE[i])) + (invXX[2+(i*2),2+(i*2)]*np.cos(ACROPHASE[i])**2)) / AMPLITUDE[i])
			SE_AMPLITUDE.append(sigma * np.sqrt((invXX[(1+(i*2)),1+(i*2)]*np.cos(ACROPHASE[i])**2) - (2*invXX[1+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])*np.cos(ACROPHASE[i])) + (invXX[2+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])**2)))

			ACROPHASE = np.array(ACROPHASE)
			if rand_array is None:
				ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)] = -ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)]
				ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)] = (-1*np.pi) + ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)]
				ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)] = (-1*np.pi) - ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)]
				ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)] = (-2*np.pi) + ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)]
			# t values
			tAMPLITUDE.append(np.divide(AMPLITUDE[i], SE_AMPLITUDE[i]))
			tACROPHASE.append(np.divide(1.0, SE_ACROPHASE[i]))

		# Do not output R-squared during permutations testing.
		if rand_array is None:
			R2 = 1 - (SS_Residuals/SS_Total)
		else:
			R2 = None
		return R2, MESOR, SE_MESOR, np.array(AMPLITUDE), np.array(SE_AMPLITUDE), np.array(ACROPHASE), np.array(SE_ACROPHASE), Fmodel, tMESOR, np.abs(tAMPLITUDE), np.abs(tACROPHASE), np.array(tEXOG)


def dummy_code(variable, iscontinous = False, demean = True):
	"""
	Dummy codes a variable
	
	Parameters
	----------
	variable : array
		1D array variable of any type 

	Returns
	---------
	dummy_vars : array
		dummy coded array of shape [(# subjects), (unique variables - 1)]
	
	"""
	if iscontinous:
		if demean:
			dummy_vars = variable - np.mean(variable,0)
		else:
			dummy_vars = variable
	else:
		unique_vars = np.unique(variable)
		dummy_vars = []
		for var in unique_vars:
			temp_var = np.zeros((len(variable)))
			temp_var[variable == var] = 1
			dummy_vars.append(temp_var)
		dummy_vars = np.array(dummy_vars)[1:] # remove the first column as reference variable
		dummy_vars = np.squeeze(dummy_vars).astype(int).T
		if demean:
			dummy_vars = dummy_vars - np.mean(dummy_vars,0)
	return dummy_vars

def dummy_code_cosine(time, period = 24.0):
	"""
	Dummy codes a time variable into a cosine
	C1 = (2.0*Pi*time)/period)
	C2 = sin(2.0*Pi*time)/period)

	Parameters
	----------
	time : array
		1D array variable of any type 
	period : float
		Defined period (i.e., for one entire cycle) for the time variable

	Returns
	---------
	cosine_curve : array
		2D array cos and sine curves 
	
	"""

	cosine_curve = np.cos(np.divide(2.0*np.pi*time, period))
	cosine_curve = np.column_stack((cosine_curve,np.sin(np.divide(2.0*np.pi*time, period))))
	return cosine_curve


def column_product(arr1, arr2):
	"""
	Multiply two dummy codes arrays
	
	Parameters
	----------
	arr1 : array
		2D array variable dummy coded array (nlength, nvars)

	arr2 : array
		2D array variable dummy coded array (nlength, nvars)

	Returns
	---------
	prod_arr : array
		dummy coded array [nlength, nvars(arr1)*nvars(arr2)]
	
	"""
	l1 = len(arr1)
	l2 = len(arr2)
	if l1 == l2:
		arr1 = np.array(arr1)
		arr2 = np.array(arr2)
		prod_arr = []
		if arr1.ndim == 1:
			prod_arr = (arr1*arr2.T).T
		elif arr2.ndim == 1:
			prod_arr = (arr2*arr1.T).T
		else:
			for i in range(arr1.shape[1]):
				prod_arr.append((arr1[:,i]*arr2.T).T)
			prod_arr = np.array(prod_arr)
			if prod_arr.ndim == 3:
				prod_arr = np.concatenate(prod_arr, axis=1)
		prod_arr[prod_arr==0]=0
		return prod_arr
	else:
		print("Error: arrays must be of same length")
		quit()

def stack_ones(arr):
	"""
	Add a column of ones to an array
	
	Parameters
	----------
	arr : array

	Returns
	---------
	arr : array
		array with a column of ones
	
	"""
	return np.column_stack([np.ones(len(arr)),arr])

def calc_indirect(ta, tb, alg = "aroian"):
	"""
	Calculates the indirect effect of simple mediation
	
	Parameters
	----------
	ta : array
		T-values from Path A
	tb : array
		T-values from Path B
	alg : string
		The algorithm used for mediation (default = "aroian") 

	Returns
	---------
	SobelZ : array
		Sobel-Z statistics of the indirect effect.
	
	"""
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

def check_blocks(block_list):
	"""
	Checks if blocks are equal sizes for permutation testing.
	
	Parameters
	----------
	block_list : array
	
	Returns
	-------
	is_equal_sizes : bool
	"""
	
	unique_blocks = np.unique(block_list)
	block_sizes = []
	for block in unique_blocks:
		block_sizes.append(len(block_list[block_list == block]))
	is_equal_sizes = all(x==block_sizes[0] for x in list(block_sizes))
	if is_equal_sizes == False:
		print("Warning: blocks are not equal. Swaping with only occur within blocks, but not among blocks.")
	return is_equal_sizes

def rand_blocks(block_list, is_equal_sizes):
	"""
	Output permutation index array based on blocks.
	
	Parameters
	----------
	block_list : array
	is_equal_sizes : bool
	
	Returns
	-------
	rand_array : array
	"""
	
	indexer = np.array(range(len(block_list)))
	randindex = []
	if is_equal_sizes is True:
		for block in np.random.permutation(list(np.unique(block_list))):
			randindex.append(np.random.permutation(indexer[block_list==block]))
		rand_array = np.concatenate(np.array(randindex))
	else:
		for block in np.unique(block_list):
			randindex.append(np.random.permutation(indexer[block_list==block]))
		rand_array = np.concatenate(np.array(randindex))
	return rand_array

def lm_residuals(endog, exog):
	"""
	"""
	if exog.ndim == 1:
		exog = stack_ones(exog)
	if np.mean(exog[:,0]) != 1:
		exog = stack_ones(exog)
	a = cy_lin_lstsqr_mat(exog,endog)
	endog = endog - np.dot(exog,a)
	return endog


# GLM COSINOR V2

def load_vars(pdCSV, variables, exog = [], names = [], demean_flag = True):
	if len(variables) % 2 == 1:
		print("Error: each input must be followed by data type. e.g., -glm age c sex d site d (d = discrete, c = continous)")
	num_exog = int(len(variables) / 2)
	for i in range(num_exog):
		j = i * 2 
		k = j + 1
		if variables[k] == 'c':
			print("Coding %s as continous variable" % variables[j])
			temp = dummy_code(np.array(pdCSV[variables[j]]), iscontinous = True, demean = demean_flag)
			temp = temp[:,np.newaxis]
			exog.append(temp)
		elif variables[k] == 'd':
			print("Coding %s as discrete variable" % variables[j])
			temp = dummy_code(np.array(pdCSV[variables[j]]), iscontinous = False, demean = demean_flag)
			if temp.ndim == 1:
				temp = temp[:,np.newaxis]
			exog.append(temp)
		else:
			print("Error: variable type is not understood")
		names.append(variables[j])
	return (exog, names)


def load_interactions(intvariables, varnames = [], exog = [], covarnames = [], covars = []):
	for int_terms in intvariables:
		interaction_vars = int_terms.split("*")
		int_name = ".X.".join(interaction_vars)
		if interaction_vars[0] in varnames:
			for i, scale_var in enumerate(interaction_vars):
				if i == 0:
					int_temp = exog[varnames.index(interaction_vars[i])]
				else:
					int_temp = column_product(int_temp, exog[varnames.index(interaction_vars[i])])
			exog.append(int_temp)
			varnames.append(int_name)
		elif interaction_vars[0] in covarnames:
			for i, scale_var in enumerate(interaction_vars):
				if i == 0:
					int_temp = covars[covarnames.index(interaction_vars[i])]
				else:
					int_temp = column_product(int_temp, covars[covarnames.index(interaction_vars[i])])
			covars.append(int_temp)
			covarnames.append(int_name)
		else:
			print("Error: interaction variables must be contained in -glm or -c")
	return (varnames, exog, covarnames, covars)




# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3991883/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3663600/
def glm_cosinor_fstats(endog, time_variable, period, iterator, exog = None, interaction = None, covars = None, blocking = None, randomise = True):
	"""
	COSINOR model using GLM
	
	Parameters
	----------
	endog : array
		Endogenous (dependent) variable array (Nsubjects, Nvariables)
	time_variable : array
		Time variable [0-23.99] (Nsubjects).
	period : array
		Period(s) as an array of floats for cosinor model.
	iterator : int
		interator variable for parallel processing
	exog : array
		Exogenous (independent) dummy coded variables
		exog is an array of arrays (Nvariables, Nsubjects, Kvariable).
	exog : array
		InteractingExogenous (independent) dummy coded variables
		exog is an array of arrays (Nvariables, Nsubjects, Kvariable).
	dmy_covariates : array
		Dummy coded array of covariates of no interest.
	init_covars : array
		Dummy coded array of covariates for two-step regression.
	rand_array : array
		randomized array for permutations (Nsubjects).

	Returns
	---------
	To-do
	"""


	n = len(time_variable)
	# Check that endog has two dimensions
	if endog.ndim == 1:
		endog = endog.reshape(len(endog),1)

	if randomise:
		rand_array = np.random.permutation(list(range(n)))
		endog = endog[rand_array]

	period = np.array(period)

	exog_model = dummy_code_cosine(time_variable, period)

	other_models = []
	for per in period:
		other_models.append(period[period!=per])
	other_models = np.array(other_models)

	k = exog_model.shape[1]
	DF_Between = k - 1 # aka df model
	DF_Within = n - k # aka df residuals

	SS_Total = np.sum((endog - np.mean(endog,0))**2,0)
	SS_Residuals = cy_lin_lstsqr_mat_residual(exog_model, endog)[1]

	SS_Between = SS_Total - SS_Residuals
	MS_Residuals = (SS_Residuals/ DF_Within)
	Fmodel = (SS_Between/DF_Between) / MS_Residuals



	n = endog.shape[0]
	# add cosinor terms
	num_period = len(period)
	exog_vars = np.ones((n))
	for i in range(num_period):
		exog_vars = np.column_stack((exog_vars,np.cos(np.divide(2.0*np.pi*time_var, period[i]))))
		exog_vars = np.column_stack((exog_vars,np.sin(np.divide(2.0*np.pi*time_var, period[i]))))



	if interaction_var is not None:
		for i in range(num_period):
			exog_vars = np.column_stack((exog_vars, exog_vars[i+1] * interaction_var))


	kvars = []
	# add other exogenous variables to the model (currently not implemented)
	if exog is not None:
		for var in exog:
			var = np.array(var)
			if var.ndim == 1:
				kvars.append((3))
			else:
				kvars.append((var.shape[1]))
			exog_vars = np.column_stack((exog_vars,var))

	# add covariates (i.e., exogenous variables that will not be outputed)
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covariates))
	exog_vars = np.array(exog_vars)

	if rand_array is not None:
		exog_vars = exog_vars[rand_array]

	# calculate model fit (Fmodel and R-sqr)
	k = exog_vars.shape[1]
	DF_Between = k - 1 # aka df model
	DF_Within = n - k # aka df residuals
	#DF_Total = n - 1

	a, SS_Residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog)
	if output_fit_only:
		AMPLITUDE = []
		ACROPHASE = []
		MESOR = a[0]
		for i in range(num_period):
			# beta, gamma
			AMPLITUDE.append(np.sqrt((a[1+(i*2),:]**2) + (a[2+(i*2),:]**2)))
			# Acrophase calculation
			if i == 0: # awful hack
				ACROPHASE = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				ACROPHASE = ACROPHASE[np.newaxis,:]
			else:
				temp_acro = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				temp_acro = temp_acro[np.newaxis,:]
				ACROPHASE = np.append(ACROPHASE,temp_acro, axis=0)
			ACROPHASE = np.array(ACROPHASE)
			ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)] = -ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)]
			ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)] = (-1*np.pi) + ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)]
			ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)] = (-1*np.pi) - ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)]
			ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)] = (-2*np.pi) + ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)]
		return MESOR, np.array(AMPLITUDE), np.array(ACROPHASE)
	else:
		SS_Total = np.sum((endog - np.mean(endog,0))**2,0)
		SS_Between = SS_Total - SS_Residuals
		MS_Residuals = (SS_Residuals / DF_Within)
		Fmodel = (SS_Between/DF_Between) / MS_Residuals
		# Calculates sigma sqr and T-value (intercept) for MESOR
		sigma = np.sqrt(SS_Residuals / DF_Within)
		invXX = np.linalg.inv(np.dot(exog_vars.T, exog_vars))

		if (calc_MESOR) or (exog is not None):
			if endog.ndim == 1:
				se = np.sqrt(np.diag(sigma * sigma * invXX))
				Tvalues = a / se
				MESOR = a[0]
				tMESOR = Tvalues[0]
				SE_MESOR = se[0]
				a = a[:, np.newaxis]
			else:
				num_depv = endog.shape[1]
				se = se_of_slope(num_depv,invXX,sigma**2,k)
				Tvalues = a / se
				MESOR = a[0,:]
				tMESOR = Tvalues[0,:]
				SE_MESOR = se[0,:]
			if exog is not None:
				tEXOG = Tvalues[(3+(2*(num_period-1))):,:]
			else:
				tEXOG = None
		else:
			MESOR = tMESOR = SE_MESOR = tEXOG = None

		AMPLITUDE = []
		ACROPHASE = []
		SE_ACROPHASE = []
		SE_AMPLITUDE = []
		tAMPLITUDE = []
		tACROPHASE = []
		
		for i in range(num_period):
			# beta, gamma
			AMPLITUDE.append(np.sqrt((a[1+(i*2),:]**2) + (a[2+(i*2),:]**2)))
			# Acrophase calculation
			if i == 0: # awful hack
				ACROPHASE = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				ACROPHASE = ACROPHASE[np.newaxis,:]
			else:
				temp_acro = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				temp_acro = temp_acro[np.newaxis,:]
				ACROPHASE = np.append(ACROPHASE,temp_acro, axis=0)

			# standard errors from error propagation
			SE_ACROPHASE.append(sigma * np.sqrt((invXX[(1+(i*2)),1+(i*2)]*np.sin(ACROPHASE[i])**2) + (2*invXX[1+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])*np.cos(ACROPHASE[i])) + (invXX[2+(i*2),2+(i*2)]*np.cos(ACROPHASE[i])**2)) / AMPLITUDE[i])
			SE_AMPLITUDE.append(sigma * np.sqrt((invXX[(1+(i*2)),1+(i*2)]*np.cos(ACROPHASE[i])**2) - (2*invXX[1+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])*np.cos(ACROPHASE[i])) + (invXX[2+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])**2)))

			ACROPHASE = np.array(ACROPHASE)
			if rand_array is None:
				ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)] = -ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)]
				ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)] = (-1*np.pi) + ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)]
				ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)] = (-1*np.pi) - ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)]
				ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)] = (-2*np.pi) + ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)]
			# t values
			tAMPLITUDE.append(np.divide(AMPLITUDE[i], SE_AMPLITUDE[i]))
			tACROPHASE.append(np.divide(1.0, SE_ACROPHASE[i]))

		# Do not output R-squared during permutations testing.
		if rand_array is None:
			R2 = 1 - (SS_Residuals/SS_Total)
		else:
			R2 = None
		return R2, MESOR, SE_MESOR, np.array(AMPLITUDE), np.array(SE_AMPLITUDE), np.array(ACROPHASE), np.array(SE_ACROPHASE), Fmodel, tMESOR, np.abs(tAMPLITUDE), np.abs(tACROPHASE), np.array(tEXOG)

