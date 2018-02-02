#!/usr/bin/env python

#    Create adjacency set using voxel adjacency, vertex mesh connectivity, or geodescic distance
#    Copyright (C) 2017 Tristram Lett

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

import numpy as np
import nibabel as nib
import os
import argparse as ap
from tfce_mediation.adjacency import compute
from tfce_mediation.pyfunc import convert_mni_object, convert_fs, convert_gifti, convert_ply
from tfce_mediation.tm_io import write_tm_filetype, read_tm_filetype

DESCRIPTION = "Create adjacency list based on geodesic distance for vertex-based TFCE. Note, 1mm, 2mm, and 3mm adjacency list have already supplied (adjacency_sets/?h_adjacency_dist_?.0_mm.npy)"

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

def create_adjac_voxel(data_index, dirtype=26): # default is 26 directions
	num_voxel = len(data_index[data_index==True])
	ind=np.where(data_index)
	dm=np.zeros((data_index.shape))
	x_dim, y_dim, z_dim=data_index.shape
	adjacency = [set([]) for i in range(num_voxel)]
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

def mergeIdenticalVertices(v, f):
	vr = np.around(v, decimals = 10)
	vrv = vr.view(v.dtype.descr * v.shape[1])
	_, idx, inv = np.unique(vrv, return_index = True, return_inverse = True)
	lkp = idx[inv]
	v_ = v[idx, :]
	f_ = np.asarray([[lkp[f[i, j]] for j in range(f.shape[1])] for i in range(f.shape[0])], dtype = np.int32)
	return v, f_

def removeNonManifoldTriangles(v, f):
	v_ = v[f]
	fn = np.cross(v_[:, 1] - v_[:, 0], v_[:, 2] - v_[:,0])
	f_ = f[np.logical_not(np.all(np.isclose(fn, 0), axis = 1)), :]
	return v, f_

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

def projNormFracThick(v, vn, t, projfrac):
	return v + (vn * t[:, None] * projfrac)

def compute_adjacency(min_dist, max_dist, step_dist, v, f, projfrac = None, t = None):
	v = v.astype(np.float32, order = "C")
	f = f.astype(np.int32, order = "C")
	v, f = mergeIdenticalVertices(v, f) # probably note necessary
	v, f = removeNonManifoldTriangles(v, f) # probably note necessary
	if t is not None:
		vn = computeNormals(v, f)
		v = projNormFracThick(v, vn, t, projfrac)
#	nib.freesurfer.io.write_geometry("%s.midthickness" % hemi, v_, f)
	thresholds = np.arange(min_dist, max_dist, step=step_dist, dtype = np.float32)
	adjacency = compute(v, f, thresholds)
	return adjacency

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):


	# output
	ogroup = ap.add_mutually_exclusive_group(required=True)
	ogroup.add_argument("-o", "--outputnpy",
		action = 'store_true',
		help = "Output adjacency set as a numpy file.") 
	ogroup.add_argument("-a", "--appendtmi",
		nargs = 1,
		help = "Append the adjacency set to a specified tmi file. i.e., -a all.surfaces.tmi", 
		metavar = ('*.tmi'))

	# input
	ap.add_argument("-t", "--datatype",
		nargs = 1,
		help = "Input file type. Note, use srf for freesurfer even if there is no extension. mni_obj refers to CIVET mni object files *.obj not waveform objects.", 
		choices = ('voxel', 'srf', 'ply', 'mni_obj', 'gii'),
		required = True)
	ap.add_argument("-i", '--input',
		nargs = '+',
		help = "Input surface object(s) or binarized images mask(s). i.e., -i lh.midthickness rh.midthickness [OR] -i mean_FA_skeleton_mask.nii.gz.",
		required = True)

	#options
	ap.add_argument("-va", "--voxeladjacency", 
		nargs = 1, 
		help = "Required with -t voxel. Set voxel adjacency to 6 or 26 direction. In general, 6 is used for volumetric data, and 26 is used for skeletonised data", 
		choices = (6,26),
		type = int,
		required = False)
	surfaceadjgroup = ap.add_mutually_exclusive_group(required=False)
	surfaceadjgroup.add_argument("-d", "--geodistance", 
		nargs = 2, 
		help = "Recommended for surfaces. Enter the [minimum distance(mm)] [maximum distance(mm)]. i.e., -d 1.0 3.0", 
		metavar = ('Float', 'Float'),
		required = False)
	ap.add_argument("-s", "--stepsize", 
		nargs = 1, 
		help = "For -d option, specify the [step size (mm)] default: %(default)s). i.e., -d 1.0", 
		metavar = ('Float'),
		default = [1.0],
		type=float)
	ap.add_argument("-p", "--projectfraction", 
		nargs = 1, 
		help = "Optional. For surface inputs, a project transformation is performed by a set fraction from white matter surface. i.e, 0.5 is midthickness surface", 
		metavar = ('Float'),
		type = float,
		required = False)
	ap.add_argument("-sa", "--setappendadj", 
		nargs = 1, 
		help = "Specify the geodesic distance to append to a tmi file. This option is strongly recommended with -a because so one adjacency set corresponds to one surface in a tmi file. i.e., -sa 3.0", 
		metavar = ('Float'),
		required = False)
	ap.add_argument("-it", "--inputthickness", 
		nargs = '+', 
		help = "Required with -p option. Input surface file for surface projection. Currently, only freesurfer is supported.", 
		metavar = ('string'),
		type = str,
		required = False)
	surfaceadjgroup.add_argument("-m", "--triangularmesh", 
		help="For surfaces, create adjacency based on triangular mesh without specifying distance (not recommended).",
		action='store_true')
	return ap

def run(opts):
	adjacency = []
	# check for tmi file first
	if opts.appendtmi:
		if not os.path.exists(opts.appendtmi[0]):
			print("Cannot find tmi file: %s" % opts.appendtmi[0])
			quit()
	if opts.datatype[0] == 'voxel':
		if not opts.voxeladjacency:
			print("-va must be specified for voxel input data")
			quit()
		for i in range(len(opts.input)):
			mask_data = nib.load(opts.input[i]).get_data()
			data_index = mask_data==1
			print("Computing adjacency for %d voxel with %d direction adjacency" % (len(data_index[data_index==True]), opts.voxeladjacency[0]))
			adjacency.append((create_adjac_voxel(data_index,dirtype=opts.voxeladjacency[0])))
	else:
		for i in range(len(opts.input)):
			if opts.datatype[0] == 'srf':
				v,f = convert_fs(str(opts.input[i]))
			if opts.datatype[0] == 'ply':
				v,f,_ = convert_ply(str(opts.input[i]))
			if opts.datatype[0] == 'mni_obj':
				v,f = convert_mni_object(str(opts.input[i]))
			if opts.datatype[0] == 'gii':
				v,f = convert_gifti(str(opts.input[i]))
			if opts.geodistance:
				min_dist=float(opts.geodistance[0])
				max_dist=float(opts.geodistance[1])
				step=float(opts.stepsize[0])
				max_dist+=step
				step_range = np.arange(min_dist, max_dist, step=step)
				if np.divide(max_dist-min_dist,step).is_integer():
					if opts.projectfraction:
						projfrac=float(opts.projectfraction[0])
						t = nib.freesurfer.read_morph_data(opts.inputthickness[i])
						temp_adjacency = compute_adjacency(min_dist, max_dist, step, v, f, projfrac = projfrac, t = t)
					else:
						temp_adjacency = compute_adjacency(min_dist, max_dist, step, v, f)
				else:
					print("The difference between max and min distance must be evenly divisible by the step size.")
					exit()
				if opts.setappendadj:
					adjacency.append((temp_adjacency[int(np.argwhere(step_range==float(opts.setappendadj[0])))]))
				else:
					if opts.appendtmi:
						print("Warning: Multiple adjacency sets are appended for each surface.")
					count = 0
					for j in step_range:
						print("Appending adjacency set at geodesic distance of %1.2f" % j)
						adjacency.append((temp_adjacency[count]))
						count += 1
			if opts.triangularmesh:
				adjacency.append((create_adjac_vertex(v, f)))

	# output adjacency sets
	if opts.outputnpy:
		for i in range(len(opts.input)):
			outname = os.path.basename(opts.input[i])
			basename = os.path.splitext(outname)[0]
			if opts.datatype[0] == 'voxel':
				outname = 'adjac_set_%d_dir%d_%s.npy' % (i,opts.voxeladjacency[0], basename)
				np.save(outname,adjacency[i])
			elif opts.setappendadj:
				outname = 'adjac_set_%d_%1.2fmm_%s.npy' % (i, float(opts.setappendadj[0]), basename)
				np.save(outname,adjacency[i])
			else:
				count = 0
				for j in step_range:
					outname = 'adjac_set_%d_%1.2fmm_%s.npy' % (i, j, basename)
					np.save(outname,adjacency[count])
					count += 1

	if opts.appendtmi:
		outname = opts.appendtmi[0]
		_, image_array, masking_array, maskname, affine_array, vertex_array, face_array, surfname, adjacency_array, tmi_history, subjectids = read_tm_filetype(outname, verbose=True)
		for i in range(len(opts.input)):
			adjacency_array.append((adjacency[i]))
		write_tm_filetype(outname, image_array = image_array[0], masking_array=masking_array, maskname=maskname, affine_array=affine_array, vertex_array=vertex_array, face_array=face_array, surfname=surfname, checkname=False, tmi_history=tmi_history, adjacency_array = adjacency_array)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
