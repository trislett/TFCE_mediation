#!/usr/bin/env python

#    Various functions for I/O functions for TFCE_mediation
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

from __future__ import division

import os
import sys
import nibabel as nib
import numpy as np
from tfce_mediation.pyfunc import check_outname

def tm_filetype_version:
	version = '0.1'
	return version

def savemgh_v2(imgdata, index, imagename, affine=None):
	outdata = imgdata.astype(np.float32, order = "C")
	if imgdata.ndim == 2:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2],imgdata.shape[1]))
	elif imgdata.ndim == 1:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2]))
	else:
		print 'error'
	imgout[index]=outdata
	nib.save(nib.freesurfer.mghformat.MGHImage(imgout.astype(np.float32, order = "C"),affine=affine),imagename)


def savenifti_v2(imgdata, index, imagename, affine=None):
	outdata = imgdata.astype(np.float32, order = "C")
	if imgdata.ndim == 2:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2],imgdata.shape[1]))
	elif imgdata.ndim == 1:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2]))
	else:
		print 'error'
	imgout[index]=outdata
	nib.save(nib.Nifti1Image(imgout.astype(np.float32, order = "C"),affine=affine),imagename)

def write_tm_filetype(imgdata=[], outname, surface_object = 'unknown', subjectid = 'unknown', hemisphere = 'unknown', imgtype = 'unknown', output_binary = True, mask_index=[], affine=[], vertices=[], faces=[]):
	num_mask = 0
	num_object = 0
	num_affine = 0
	if not mask_index==[]:
		if mask_index.ndim==3:
			num_mask=1
		elif mask_index.ndim==4:
			num_mask=int(mask_index.shape[0])
			hemisphere = 'both'
		else:
			print "Error mask dimension are not understood"
	if not vertices==[]:
		if vertices.ndim==2:
			num_object=1
		elif vertices.ndim==3:
			num_object=int(vertices.shape[0])
		else:
			print "Error surface object dimension are not understood."
	if not affine==[]:
		if affine.ndim==2:
			num_affine=1
		elif affine.ndim==3:
			num_affine=int(vertices.shape[0])
		else:
			print "Error affine dimension are not understood."

	# write array shape
	if imgdata.ndim == 1:
		nvert=len(imgdata)
		nsub=1
	else:
		nvert=imgdata.shape[0]
		nsub=imgdata.shape[1]
	if not outname.endswith('ascii'):
		if output_binary:
			outname += '.tmi'
		else:
			outname += '.ascii.tmi'
	outname=check_outname(outname)
	with open(outname, "wb") as o:
		o.write("tmi\n")
		if output_binary:
			o.write("format binary_%s_endian %s\n" % ( sys.byteorder, tm_filetype_version() ) )
		else:
			o.write("format ascii %s\n" % tm_filetype_version() )
		o.write("comment made with TFCE_mediation\n")
		if not imgdata==[]:
			o.write("element data_array\n")
			o.write("dtype float32\n")
			o.write("nbytes %d\n" % imgdata.nbytes)
			o.write("numV %d\n" % nvert)
			o.write("numS %d\n" % nsub)
		if num_mask==1:
			o.write("element masking_array\n")
			o.write("dtype uint8\n")
			o.write("nbytes %d\n" % mask_index.nbytes)
			o.write("nmasked %d\n" % len(mask_index[mask_index==True]))
			o.write("maskshape %d %d %d\n" % (mask_index.shape[0],mask_index.shape[1],mask_index.shape[2]))
		if num_mask>1:
			for i in range(num_mask):
				o.write("element masking_array\n")
				o.write("dtype uint8\n") # for binarized masks
				o.write("nbytes %d\n" % mask_index[i].nbytes)
				o.write("nmasked %d\n" % len(mask_index[i][mask_index[i]==True]))
				o.write("maskshape %d %d %d\n" % (mask_index[i].shape[0],mask_index[i].shape[1],mask_index[i].shape[2]))
		o.write("surface_obj %s\n" % surface_object)
		o.write("subject_id %s\n" % subjectid)
		o.write("hemisphere %s\n" % hemisphere) #lh, rh, bh
		o.write("image_type %s\n" % imgtype)

		if num_affine==1:
			affine = affine.astype('float32') # make sure it is a float32
			o.write("element affine\n")
			o.write("dtype float32\n")
			o.write("nbytes %d\n" % affine.nbytes)
			o.write("affineshape %d %d\n" % (affine.shape[0], affine.shape[1]) )
		if num_affine>1:
			for i in range(num_object):
				affine[i] = affine[i].astype('float32') # make sure it is a float32
				o.write("element vertex\n")
				o.write("dtype int32\n")
				o.write("nbytes %d\n" % affine[i].nbytes)
				o.write("affineshape %d %d\n" % (affine[i].shape[0], affine[i].shape[1]) )

		if num_object==1:
			vertices = vertices.astype('float32') # make sure it is a float32
			faces = faces.astype('uint32')

			o.write("element vertex\n")
			o.write("dtype float32\n")
			o.write("nbytes %d\n" % vertices.nbytes)
			o.write("vertexshape %d %d\n" % (vertices.shape[0], vertices.shape[1]) )
			o.write("element face\n")
			o.write("dtype uint32\n")
			o.write("nbytes %d\n" % faces.nbytes)
			o.write("faceshape %d %d\n" % (faces.shape[0], faces.shape[1]) )

		if num_object>1:
			for i in range(num_object):
				vertices[i] = vertices[i].astype('float32') # make sure it is a float32
				faces[i] = faces[i].astype('uint32')

				o.write("element vertex\n")
				o.write("dtype float32\n")
				o.write("nbytes %d\n" % vertices[i].nbytes)
				o.write("vertexshape %d %d\n" % (vertices[i].shape[0], vertices[i].shape[1]) )

				o.write("element face\n")
				o.write("dtype uint32\n")
				o.write("nbytes %d\n" % faces[i].nbytes)
				o.write("faceshape %d %d\n" % (faces[i].shape[0], faces[i].shape[1]))
		o.write("end_header\n")

		if output_binary:
			imgdata = np.array(imgdata.T, dtype='float32') # transpose to reduce file size
			imgdata.tofile(o)
			if num_mask==1: # inefficient
				binarymask = mask_index * 1
				binarymask = np.array(binarymask.T, dtype=np.uint8)
				binarymask.tofile(o)
			if num_mask>1:
				for j in range(num_mask):
					binarymask = mask_index[j] * 1
					binarymask = np.array(binarymask.T, dtype=np.uint8)
					binarymask.tofile(o)
			if num_affine==1: # inefficient
				affine = np.array(affine.T, dtype='float32')
				affine.tofile(o)
			if num_affine>1:
				for j in range(num_affine):
					outaffine = np.array(affine[j].T, dtype='float32')
					outaffine.tofile(o)
			if num_object==1: # inefficient
				vertices = np.array(vertices.T, dtype='float32')
				vertices.tofile(o)
				faces = np.array(faces.T, dtype='uint32')
				faces.tofile(o)
			if num_object>1:
				for j in range(num_mask):
					outv = np.array(vertices[j].T, dtype='float32')
					outv.tofile(o)
					outf = np.array(faces[j].T, dtype='uint32')
					outf.tofile(o)
		else:
			np.savetxt(o,imgdata.astype(np.float32))
			if num_mask==1:
				binarymask = mask_index * 1
				binarymask = np.array(binarymask, dtype=np.uint8)
				x, y, z = np.ma.nonzero(binarymask)
				for i in xrange(len(x)):
					o.write("%d %d %d\n" % (int(x[i]), int(y[i]), int(z[i]) ) )
			if num_mask>1:
				for j in range(num_mask):
					binarymask = mask_index[j] * 1
					binarymask = np.array(binarymask, dtype=np.uint8)
					x, y, z = np.ma.nonzero(binarymask)
					for i in xrange(len(x)):
						o.write("%d %d %d\n" % (int(x[i]), int(y[i]), int(z[i]) ) )
			if num_affine==1: # inefficient
				affine = np.array(affine)
				np.savetxt(o,affine.astype(np.float32))
			if num_affine>1:
				for j in range(num_affine):
					outaffine = np.array(affine[j])
					np.savetxt(o,outaffine.astype(np.float32))
			if num_object==1: # inefficient
				for i in xrange(len(vertices)):
					o.write("%1.6f %1.6f %1.6f\n" % (vertices[i,0], vertices[i,0], vertices[i,0] ) )
				for j in xrange(len(faces)):
					o.write("%d %d %d\n" % (int(faces[j,0]), int(faces[j,1]), int(faces[j,2]) ) )
			if num_object>1:
				for k in range(num_object):
					for i in xrange(len(vertices[k])):
						o.write("%1.6f %1.6f %1.6f\n" % (vertices[k][i,0], vertices[k][i,1], vertices[k][i,2] ) )
					for j in xrange(len(faces[k])):
						o.write("%d %d %d\n" % (int(faces[k][j,0]), int(faces[k][j,1]), int(faces[k][j,2]) ) )
		o.close()

def read_tm_filetype(tm_file):
	#getfilesize
	filesize = os.stat(tm_file).st_size
	#read first line
	obj = open(tm_file)
	reader = obj.readline().strip().split()
	firstword=reader[0]
	#declare variables
	element = []
	element_dtype = []
	element_nbyte = []
	element_nmasked = []
	masking_array = []
	maskshape = []
	vertexshape = []
	faceshape = []
	affineshape = []
	array_read = []
	o_imgarray = []
	o_masking_array = []
	o_vertex = []
	o_face = []
	o_affine = []
	maskcounter=0
	vertexcounter=0
	facecounter=0
	affinecounter=0
	if firstword != 'tmi':
		print "Error: not a TFCE_mediation image."
		quit()

	reader = obj.readline().strip().split()
	if firstword != 'format':
		print "Error: unknown reading file format"
		quit()
	else:
		tm_filetype = reader[1]
	while firstword != 'end_header':
		reader = obj.readline().strip().split()
		firstword=reader[0]
		if firstword=='element':
			element.append((reader[1]))
		if firstword=='dtype':
			element_dtype.append((reader[1]))
		if firstword=='nbytes':
			element_nbyte.append((reader[1]))
		if firstword=='numV':
			numV = int(reader[1])
		if firstword=='numS':
			numS = int(reader[1])
		if firstword=='nmasked':
			element_nmasked.append(( int(reader[1]) ))
		if firstword=='maskshape':
			maskshape.append(np.array((reader[1], reader[2], reader[3])).astype(np.int))
		if firstword=='affineshape':
			affineshape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword=='vertexshape':
			vertexshape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword=='faceshape':
			faceshape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword=='surface_obj':
			surface_obj = str(reader[1])
		if firstword=='subject_id':
			subjectid = str(reader[1])
		if firstword=='hemisphere':
			hemisphere = str(reader[1])
		if firstword=='image_type':
			imgtype = str(reader[1])
	# skip header
	position = filesize
	for i in range(len(element_nbyte)):
		position-=int(element_nbyte[i])
	# readdata

	if tm_filetype == 'binary_little_endian':
		for e in range(len(element)):
			print "reading %s" % str(element[e])
			obj.seek(position)
			print position
			array_read.append((np.fromfile(obj, dtype=element_dtype[e])))
			position += int(element_nbyte[e])
			#reshape arrays
			if str(element[e]) == 'data_array':
				o_imgarray.append(np.array(array_read[e][:numV*numS]).reshape(numS,numV).T)
			if str(element[e]) == 'masking_array':
				masktemp = np.array(array_read[e][:(maskshape[maskcounter][2]*maskshape[maskcounter][1]*maskshape[maskcounter][0])]).reshape(maskshape[maskcounter][2],maskshape[maskcounter][1],maskshape[maskcounter][0]).T
				o_masking_array.append((np.array(masktemp, dtype=bool) ))
				maskcounter+=1
			if str(element[e]) == 'affine':
				o_affine.append(np.array(array_read[e][:affineshape[affinecounter][1]*affineshape[affinecounter][0]]).reshape(affineshape[affinecounter][1],affineshape[affinecounter][0]).T)
				affinecounter+=1
			if str(element[e]) == 'vertex':
				o_vertex.append(np.array(array_read[e][:vertexshape[vertexcounter][1]*vertexshape[vertexcounter][0]]).reshape(vertexshape[vertexcounter][1],vertexshape[vertexcounter][0]).T)
				vertexcounter+=1
			if str(element[e]) == 'face':
				o_face.append(np.array(array_read[e][:faceshape[facecounter][1]*faceshape[facecounter][0]]).reshape(faceshape[facecounter][1],faceshape[facecounter][0]).T)
				facecounter+=1

	elif tm_filetype == 'ascii':
		for e in range(len(element)):
			if str(element[e]) == 'data_array':
				img_data = np.zeros((numV, numS))
				for i in range(int(numV)):
					img_data[i] = np.array(obj.readline().strip().split(), dtype = 'float32')
				o_imgarray.append(( np.array(img_data, dtype = 'float32') ))
			if str(element[e]) == 'masking_array':
					for k in range(element_nmasked[maskcounter]):
						masking_array.append((np.array(obj.readline().strip().split()).astype(np.int32)))
					masking_array = np.array(masking_array)
					outmask = np.zeros((maskshape[maskcounter]), dtype=np.int)
					outmask[masking_array[:,0],masking_array[:,1],masking_array[:,2]]=1
					o_masking_array.append(( np.array(outmask, dtype=bool) ))
					maskcounter+=1
					masking_array=[]
			if str(element[e]) == 'affine':
				temparray = []
				for k in range(int(affineshape[affinecounter][0])):
					temparray.append((np.array(obj.readline().strip().split()).astype('float32')))
				o_affine.append((np.array(temparray, dtype='float32')))
				affinecounter+=1
			if str(element[e]) == 'vertex':
				temparray = []
				for k in range(int(vertexshape[vertexcounter][0])):
					temparray.append((np.array(obj.readline().strip().split()).astype('float32')))
				o_vertex.append((np.array(temparray, dtype='float32')))
				vertexcounter+=1
			if str(element[e]) == 'face':
				temparray = []
				for k in range(int(faceshape[facecounter][0])):
					temparray.append((np.array(obj.readline().strip().split()).astype('int32')))
				o_face.append(( np.array(temparray, dtype='int32') ))
				facecounter+=1
	else:
		print "Error unknown filetype: %s" % tm_filetype
	return(o_imgarray,numV,numS,surface_obj,subjectid,hemisphere,imgtype, o_masking_array, o_affine, o_vertex, o_face)
