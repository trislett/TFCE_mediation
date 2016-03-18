#!/usr/bin/python

import numpy as np
import nibabel as nib
import os
import sys
from cython.Adjacency import compute

def mergeIdenticalVertices(v, f):
  vr = numpy.around(v, decimals = 10)
  vrv = vr.view(v.dtype.descr * v.shape[1])
  _, idx, inv = numpy.unique(vrv, return_index = True, return_inverse = True)

  lkp = idx[inv]

  v_ = v[idx, :]
  f_ = numpy.asarray([[lkp[f[i, j]] for j in xrange(f.shape[1])] for i in xrange(f.shape[0])], dtype = numpy.int32)

  return v, f_

def removeNonManifoldTriangles(v, f):
  v_ = v[f]
  fn = numpy.cross(v_[:, 1] - v_[:, 0], v_[:, 2] - v_[:,0])

  f_ = f[numpy.logical_not(numpy.all(numpy.isclose(fn, 0), axis = 1)), :]

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

def compute_adjacency(hemi, min_dist, max_dist, projfrac,step_dist):
	v, f = nib.freesurfer.read_geometry("%s/fsaverage/surf/%s.white" % ((os.environ["SUBJECTS_DIR"]),hemi))
	v = v.astype(np.float32, order = "C")
	f = f.astype(np.int32, order = "C")

	v, f = Manifold.mergeIdenticalVertices(v, f)
	v, f = Manifold.removeNonManifoldTriangles(v, f)

	vn = computeNormals(v, f)

	t = nib.freesurfer.read_morph_data("%s/fsaverage/surf/%s.thickness" % ((os.environ["SUBJECTS_DIR"]),hemi))

	v_ = projNormFracThick(v, vn, t, projfrac) # project to midthickness

	nib.freesurfer.io.write_geometry("%s.midthickness" % hemi, v_, f)
	
	thresholds = np.arange(min_dist, max_dist, step=step_dist, dtype = np.float32)
	adjacency = compute(v_, f, thresholds)
	count = 0
	for i in np.arange(min_dist, max_dist, step=step):
		np.save("%s_adjacency_dist_%.1f_mm" % (hemi,i),adjacency[count])
		count += 1

if len(sys.argv) < 3:
	print "Usage: %s [Minimun distance] [Maximum distance]] optional: [step size] [projection fraction]" % (str(sys.argv[0]))
	print "By defaults step size = 1mm and projection fraction = 0.5 (i.e. midthickness)"
else:
	if len(sys.argv) == 3:
		min_dist=float(sys.argv[1])
		max_dist=float(sys.argv[2])+1
		step = 1.0
		projfrac=0.5

	elif len(sys.argv) == 4:
		min_dist=float(sys.argv[1])
		max_dist=float(sys.argv[2])+1
		step=float(sys.argv[3])
		projfrac=0.5
	elif len(sys.argv) == 5:
		min_dist=float(sys.argv[1])
		max_dist=float(sys.argv[2])+1
		step=float(sys.argv[3])
		projfrac=float(sys.argv[3])
	elif len(sys.argv) > 5:
		print "Too many arguments"
		exit()
	compute_adjacency('lh', min_dist, max_dist,projfrac,step)
	compute_adjacency('rh', min_dist, max_dist,projfrac,step)
