#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import argparse as ap

DESCRIPTION = """
Conversion of surfaces (freesurfer, gifti *.gii, mni *.obj) to freesurfer surface (as well as waveform obj or Stl triangular mesh) for analysis with TFCE_mediation.
"""

def check_outname(outname):
	if os.path.exists(outname):
		outpath,outname = os.path.split(outname)
		if not outpath:
			outname = ("new_%s" % outname)
		else:
			outname = ("%s/new_%s" % (outdir,outname))
		print "Output file aleady exists. Renaming output file to %s" % outname
	return outname

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

def convert_mni_object(obj_file):
	# adapted from Jon Pipitone's script https://gist.github.com/pipitone/8687804
	obj = open(obj_file)
	_, _, _, _, _, _, numpoints = obj.readline().strip().split()
	numpoints = int(numpoints)
	vertices=[]
	normals=[]
	triangles=[]

	for i in range(numpoints):
		x, y, z = map(float,obj.readline().strip().split()) 
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
	points = map(int, "".join(obj.readlines()).strip().split())
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

def save_waveform(v,f, outname):
	outname=check_outname(outname)
	with open(outname, "a") as o:
		for i in xrange(len(v)):
			o.write("v %1.6f %1.6f %1.6f\n" % (v[i,0],v[i,1], v[i,2]) )
		for j in xrange(len(f)):
			o.write("f %d %d %d\n" % (f[j,0],f[j,1], f[j,2]) )
		o.close()

def save_stl(v,f, outname):
	outname=check_outname(outname)
	v = np.array(v, dtype=np.float32, order = "C")
	f = np.array(f, dtype=np.int32, order = "C")
#	vn = computeNormals(v, f)
	with open(outname, "a") as o:
		o.write("solid surface\n")
		for index_f in f:
			p1 = v[index_f[0]]
			p2 = v[index_f[1]]
			p3 = v[index_f[2]]
			o.write("\tfacet normal 0 0 0\n")
			o.write("\t\tvertex %1.6e %1.6e %1.6e\n" % (p1[0],p1[1],p1[2]))
			o.write("\t\tvertex %1.6e %1.6e %1.6e\n" % (p2[0],p2[1],p2[2]))
			o.write("\t\tvertex %1.6e %1.6e %1.6e\n" % (p3[0],p3[1],p3[2]))
			o.write("\tendloop\n")
		o.write("endfacet")
		o.close()

def save_fs(v,f, outname):
	outname=check_outname(outname)
	nib.freesurfer.io.write_geometry(outname, v, f)

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	#input type
	igroup = ap.add_mutually_exclusive_group(required=True)
	igroup.add_argument("-i_fs", "--inputfreesurfer",
		help="Input a freesurfer surface (e.g., -i_fs $SUBJECTS_DIR/fsaverage/surf/lh.midthickness)", 
		nargs=1, 
		metavar=('*'))
	igroup.add_argument("-i_gifti", "--inputgifti",
		help="Input a gifti surface file (e.g., --i_gifti average.surf.gii)", 
		nargs=1, 
		metavar=('*.surf.gii'))
	igroup.add_argument("-i_mni", "--inputmniobj",
		help="Input a MNI object file (e.g., --i_mni l_hemi.obj)", 
		nargs=1, 
		metavar=('*.obj'))
	ogroup = ap.add_mutually_exclusive_group(required=True)
	ogroup.add_argument("-o_fs", "--outputfreesurfer",
		help="Output file name for freesurfer surface (e.g., -o_fs lh.32k.midthickness)", 
		nargs=1, 
		metavar=('*'))
	ogroup.add_argument("-o_obj", "--outputwaveform",
		help="Output file name for waveform object file for visualization with blender (or any other 3D viewer). This is NOT an MNI object file.", 
		nargs=1, 
		metavar=('*'))
	ogroup.add_argument("-o_stl", "--outputstl",
		help="Output file name for STereoLithography object file for visualization with blender (or any other 3D viewer).", 
		nargs=1, 
		metavar=('*'))
	return ap

def run(opts):
	#input
	if opts.inputfreesurfer:
		v,f = convert_fs(str(opts.inputfreesurfer[0]))
	if opts.inputgifti:
		v,f = convert_gifti(str(opts.inputgifti[0]))
	if opts.inputmniobj:
		v,f = convert_mni_object(str(opts.inputmniobj[0]))
	#output
	if opts.outputfreesurfer:
		save_fs(v,f, opts.outputfreesurfer[0])
	if opts.outputwaveform:
		save_waveform(v,f, opts.outputwaveform[0])
	if opts.outputstl:
		save_stl(v,f, opts.outputstl[0])

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)


