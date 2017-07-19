#!/usr/bin/env python

from __future__ import division
import os
import numpy as np
import argparse as ap
from joblib import Parallel, delayed
from joblib import load, dump
import nibabel as nib

from tfce_mediation.adjacency import compute_distance_parallel


DESCRIPTION = """
This script create the distance list for each vertex that is necessary for geodesic FWHM smoothing. A maximum distance threshold must be set to limit the size of the distance list. The default value is 9mm which works well for 3mm smoothing. The number of core for parallel processing must also be entered
"""

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	ap.add_argument("-t", "--threshold",
		help="The threshold distance from each vertex for storing", 
		nargs=1,
		default=[9.0],
		metavar=('float'))
	ap.add_argument("-n", "--numcores",  
		help="The number of cores used for parallel processing", 
		nargs=1,
		metavar=('int'),
		required=True)
	return ap

def mergeIdenticalVertices(v, f):
	vr = np.around(v, decimals = 10)
	vrv = vr.view(v.dtype.descr * v.shape[1])
	_, idx, inv = np.unique(vrv, return_index = True, return_inverse = True)
	lkp = idx[inv]
	v_ = v[idx, :]
	f_ = np.asarray([[lkp[f[i, j]] for j in xrange(f.shape[1])] for i in xrange(f.shape[0])], dtype = np.int32)
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

def outputDistance(v_, f, threshold,i):
	data_array = compute_distance_parallel(v_, f, threshold,i)
	return data_array

def loop_inplace_sum(arrlist):
	# assumes len(arrlist) > 0
	sumval = arrlist[0]
	for a in arrlist[1:]:
		sumval += a
	return sumval

def run(opts):
	threshold = float(opts.threshold[0])
	numcores = int(opts.numcores[0])

	for hemi in ['lh','rh']:
		v, f = nib.freesurfer.read_geometry("%s/fsaverage/surf/%s.white" % ((os.environ["SUBJECTS_DIR"]),hemi))
		v = v.astype(np.float32, order = "C")
		f = f.astype(np.int32, order = "C")
		v, f = mergeIdenticalVertices(v, f)
		v, f = removeNonManifoldTriangles(v, f)
		vn = computeNormals(v, f)
		t = nib.freesurfer.read_morph_data("%s/fsaverage/surf/%s.thickness" % ((os.environ["SUBJECTS_DIR"]),hemi))
		v_ = projNormFracThick(v, vn, t, 0.5) # project to midthickness

		filenamev = 'joblib_temp_v.mmap'
		filenamef = 'joblib_temp_f.mmap'
		if os.path.exists(filenamev): os.unlink(filenamev)
		_ = dump(v_, filenamev)
		if os.path.exists(filenamef): os.unlink(filenamef)
		_ = dump(f, filenamef)
		mapped_v_ = load(filenamev, mmap_mode='r+')
		mapped_f = load(filenamef, mmap_mode='r+')

		distance_values = Parallel(n_jobs=numcores)(delayed(outputDistance)(mapped_v_, mapped_f, threshold, i) for i in xrange(len(v_)))
		distance_values_flat = loop_inplace_sum(distance_values)
		distance_values_flat = np.asarray(distance_values_flat)
		indices = distance_values_flat[:,:2].astype(np.int32, order = "c")
		dist = distance_values_flat[:,2].astype(np.float32, order = "c")

		np.save('%s_%1.1fmm_fwhm_indices.npy' % (hemi, threshold),indices)
		np.save('%s_%1.1fmm_fwhm_distances.npy' % (hemi, threshold),dist)
		del distance_values
		os.remove(filenamev)
		os.remove(filenamef)


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
