#!/usr/bin/env python

from __future__ import division
import os
import numpy as np
import nibabel as nib
import argparse as ap

DESCRIPTION = """
Conversion of surfaces (freesurfer, gifti *.gii, mni *.obj) to freesurfer surface (as well as waveform obj or Stl triangular mesh) for analysis with TFCE_mediation. *mgh files can also be imported and converted to stanford pyl files.
"""

def check_outname(outname):
	if os.path.exists(outname):
		outpath,outname = os.path.split(outname)
		if not outpath:
			outname = ("new_%s" % outname)
		else:
			outname = ("%s/new_%s" % (outpath,outname))
		print "Output file aleady exists. Renaming output file to %s" % outname
		if os.path.exists(outname):
			print "%s also exists. Overwriting the file." % outname
			os.remove(outname)
	return outname

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
	if not outname.endswith('obj'):
		outname += '.obj'
	outname=check_outname(outname)
	with open(outname, "a") as o:
		for i in xrange(len(v)):
			o.write("v %1.6f %1.6f %1.6f\n" % (v[i,0],v[i,1], v[i,2]) )
		for j in xrange(len(f)):
			o.write("f %d %d %d\n" % (f[j,0],f[j,1], f[j,2]) )
		o.close()

def save_stl(v,f, outname):
	if not outname.endswith('stl'):
		outname += '.stl'
	outname=check_outname(outname)
	v = np.array(v, dtype=np.float32, order = "C")
	f = np.array(f, dtype=np.int32, order = "C")
	tris = v[f]
	n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
	n = normalize_v3(n)
	with open(outname, "a") as o:
		o.write("solid surface\n")
		for i in xrange(tris.shape[0]):
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
	outname=check_outname(outname)
	if not outname.endswith('srf'):
		outname += '.srf'
	nib.freesurfer.io.write_geometry(outname, v, f)

def convert_redtoyellow(threshold,img_data, baseColour=[227,218,201]):
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
	return color_array

def convert_bluetolightblue(threshold,img_data, baseColour=[227,218,201]):
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
	return color_array

def save_ply(v,f, outname, color_array=np.array([]) ):
	outname=check_outname(outname)
	if not outname.endswith('ply'):
		outname += '.ply'
	with open(outname, "a") as o:
		o.write("ply\n")
		o.write("format ascii 1.0\n")
		o.write("comment made with TFCE_mediation\n")
		o.write("element vertex %d\n" % len(v))
		if color_array.size == 0: #there's probably a better way to do this
			o.write("property float x\n")
			o.write("property float y\n")
			o.write("property float z\n")
		else:
			o.write("property float x\n")
			o.write("property float y\n")
			o.write("property float z\n")
			o.write("property uchar red\n")
			o.write("property uchar green\n")
			o.write("property uchar blue\n")
		o.write("element face %d\n" % len(f))
		o.write("property list uchar int vertex_index\n")
		o.write("end_header\n")
		if color_array.size == 0:
			for i in xrange(len(v)):
				o.write("%1.6f %1.6f %1.6f\n" % (v[i,0],v[i,1], v[i,2]) )
		else: 
			for i in xrange(len(v)):
				o.write("%1.6f %1.6f %1.6f %d %d %d\n" % (v[i,0],v[i,1], v[i,2], color_array[i,0],color_array[i,1], color_array[i,2]) )
		for j in xrange(len(f)):
			o.write("3 %d %d %d\n" % (f[j,0],f[j,1], f[j,2]) )
		o.close()

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
		help="Output file name for STereoLithography (STL) object file for visualization with blender (or any other 3D viewer).", 
		nargs=1, 
		metavar=('*'))
	ogroup.add_argument("-o_pyl", "--outputpyl",
		help="Output file name for Polygon File Format (PYL) object file for visualization with blender (or any other 3D viewer).", 
		nargs=1, 
		metavar=('*'))
	ap.add_argument("-p", "--paintsurface",
		help="Projects surface file onto a ply mesh for visualization of resutls using a 3D viewer. Must be used with -o_ply option. Input the surface file (*.mgh), the sigificance threshold (low and high), and red-yellow (r_y) or blue-lightblue (b-lb) colour schemes. Note, thresholds must be postive.", 
		nargs=4, 
		metavar=('*.mgh','float','float', 'r_y or b_lb'))
	ap.add_argument("-s", "--paintsecondsurface",
		help="Projects a second surface file onto a ply mesh for visualization of resutls using a 3D viewer. Must be used with -o_ply and -p options. Input the surface file (*.mgh), the sigificance threshold (low and high), and red-yellow (r_y) or blue-lightblue (b_lb) colour schemes. Note, thresholds must be postive.", 
		nargs=4, 
		metavar=('*.mgh','float','float', 'r_y or b_lb'))
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
	if opts.outputpyl:
		if opts.paintsurface:
			img = nib.load(opts.paintsurface[0])
			img_data = img.get_data()
			if img_data.ndim > 3:
				print "Error: input file can only contain one subject"
				exit()
			img_data = img_data[:,0,0]
			if (str(opts.paintsurface[3]) == 'r_y') or (str(opts.paintsurface[3]) == 'red-yellow'):
				out_color_array = convert_redtoyellow(np.array(( float(opts.paintsurface[1]),float(opts.paintsurface[2]) )), img_data)
			elif (str(opts.paintsurface[3]) == 'b_lb') or (str(opts.paintsurface[3]) == 'blue-lightblue'):
				out_color_array = convert_bluetolightblue(np.array(( float(opts.paintsurface[1]),float(opts.paintsurface[2]) )), img_data)
			else:
				print "Colour scheme %s does not exist" % str(opts.paintsecondsurface[3])
				exit()
			if opts.paintsecondsurface:
				img = nib.load(opts.paintsecondsurface[0])
				img_data = img.get_data()
				if img_data.ndim > 3:
					print "Error: input file can only contain one subject"
					exit()
				img_data = img_data[:,0,0]
				index = img_data > float(opts.paintsecondsurface[1])
				if (str(opts.paintsecondsurface[3]) == 'r_y') or (str(opts.paintsecondsurface[3]) == 'red-yellow'):
					out_color_array2 = convert_redtoyellow(np.array(( float(opts.paintsecondsurface[1]),float(opts.paintsecondsurface[2]) )), img_data)
				elif (str(opts.paintsecondsurface[3]) == 'b_lb') or (str(opts.paintsecondsurface[3]) == 'blue-lightblue'):
					out_color_array2 = convert_bluetolightblue(np.array(( float(opts.paintsecondsurface[1]),float(opts.paintsecondsurface[2]) )), img_data)
				else:
					print "Error: colour scheme %s does not exist" % str(opts.paintsecondsurface[3])
					exit()
			out_color_array[index,:] = out_color_array2[index,:]
			save_ply(v,f, opts.outputpyl[0], out_color_array)
		else:
			save_stl(v,f, opts.outputpyl[0])


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)


