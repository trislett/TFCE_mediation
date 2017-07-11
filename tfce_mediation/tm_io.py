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
try:
	import cPickle as pickle
except:
	import pickle
import nibabel as nib
import numpy as np
from time import gmtime, strftime
from tfce_mediation.pyfunc import check_outname, save_fs


# Helper functions
def tm_filetype_version():
	version = '0.1'
	return version


def savemgh_v2(image_array, index, imagename, affine=None):
	if not imagename.endswith('mgh'):
		imagename += '.mgh'
	outdata = image_array.astype(np.float32, order = "C")
	if image_array.ndim == 1:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2]))
		imgout[index]=outdata
	elif image_array.shape[1] > 1:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2],image_array.shape[1]))
		imgout[index]=outdata
	else:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2]))
		imgout[index]=outdata[:,0]
	nib.save(nib.freesurfer.mghformat.MGHImage(imgout.astype(np.float32, order = "C"),affine=affine),imagename)


def savenifti_v2(image_array, index, imagename, affine=None):
	if not ((imagename.endswith('nii')) or (imagename.endswith('nii.gz'))):
		imagename += '.nii.gz'
	outdata = image_array.astype(np.float32, order = "C")
	if image_array.ndim == 2:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2],image_array.shape[1]))
	elif image_array.ndim == 1:
		imgout = np.zeros((index.shape[0],index.shape[1],index.shape[2]))
	else:
		print 'error'
	imgout[index]=outdata
	nib.save(nib.Nifti1Image(imgout.astype(np.float32, order = "C"),affine=affine),imagename)

###############
#  WRITE TMI  #
###############

def write_tm_filetype(outname, subjectids = [], imgtype = [], checkname = True, output_binary = True, image_array = [], masking_array = [], maskname = [],  affine_array = [], vertex_array = [], face_array = [], surfname = [], adjacency_array = [], tmi_history = []): # NOTE: add ability to store subjectids and imgtypes
	# timestamp
	currentTime=int(strftime("%Y%m%d%H%M%S",gmtime()))
	# counters
	num_data = 0
	num_mask = 0
	num_object = 0
	num_affine = 0
	num_adjacency = 0
	# history counters
	h_mask = 0
	h_affine = 0
	h_object = 0
	h_adjacency = 0

	if not tmi_history == []:
		for i in range(len(tmi_history)):
			line = tmi_history[i].split(' ')
			if line[1] == 'mode_add':
				h_mask += int(line[4])
				h_affine += int(line[5])
				h_object += int(line[6])
				h_adjacency += int(line[7])
			elif line[1] == 'mode_sub':
				h_mask -= int(line[4])
				h_affine -= int(line[5])
				h_object -= int(line[6])
				h_adjacency -= int(line[7])
			else:
				print ("Error reading history. Mode %s is not understood. Count is reflect number of element in current file" % line[1])

	if not masking_array == []:
		masking_array=np.array(masking_array)
		if (masking_array.dtype.kind=='O') or (masking_array.ndim==4):
			num_mask=int(masking_array.shape[0])
		elif masking_array.ndim==3:
			num_mask = 1
		else:
			print "Error mask dimension are not understood"
	if not vertex_array==[]:
		vertex_array=np.array(vertex_array)
		if vertex_array.ndim==2:
			num_object = 1
		elif (vertex_array.dtype.kind == 'O') or (vertex_array.ndim==3):
			num_object=int(vertex_array.shape[0])
		else:
			print "Error surface object dimension are not understood."
	if not affine_array==[]:
		affine_array=np.array(affine_array)
		if affine_array.ndim==2:
			num_affine = 1
		elif (affine_array.dtype.kind == 'O') or (affine_array.ndim==3):
			num_affine=int(affine_array.shape[0])
		else:
			print "Error affine dimension are not understood."
	if not adjacency_array==[]:
		adjacency_array=np.array(adjacency_array)
		if (adjacency_array.dtype.kind == 'O') or (adjacency_array.ndim==2):
			num_adjacency=int(adjacency_array.shape[0])
		elif adjacency_array.ndim==1:
			num_adjacency = 1
		else:
			print "Error shape of adjacency objects are not understood."

	# write array shape
	if not image_array==[]:
		num_data = 1
		if image_array.ndim == 1:
			nvert=len(image_array)
			nsub = 1
		else:
			nvert=image_array.shape[0]
			nsub=image_array.shape[1]
	if not outname.endswith('tmi'):
		if output_binary:
			if not outname.endswith('tmi'):
				outname += '.tmi'
		else:
			outname += '.ascii.tmi'
	if checkname:
		outname=check_outname(outname)
	with open(outname, "wb") as o:
		o.write("tmi\n")
		if output_binary:
			o.write("format binary_%s_endian %s\n" % ( sys.byteorder, tm_filetype_version() ) )
		else:
			o.write("format ascii %s\n" % tm_filetype_version() )
		o.write("comment made with TFCE_mediation\n")
		if not image_array==[]:
			o.write("element data_array\n")
			o.write("dtype float32\n")
			o.write("nbytes %d\n" % image_array.astype('float32').nbytes)
			o.write("datashape %d %d\n" % (nvert,nsub))

		if num_mask>0:
			for i in range(num_mask):
				o.write("element masking_array\n")
				o.write("dtype uint8\n") # for binarized masks
				o.write("nbytes %d\n" % masking_array[i].nbytes)
				o.write("nmasked %d\n" % len(masking_array[i][masking_array[i]==True]))
				o.write("maskshape %d %d %d\n" % (masking_array[i].shape[0],masking_array[i].shape[1],masking_array[i].shape[2]))
				if not maskname==[]:
					o.write("maskname %s\n" % maskname[i])
				else:
					o.write("maskname unknown\n")

		if num_affine>0:
			for i in range(num_affine):
				o.write("element affine\n")
				o.write("dtype float32\n")
				o.write("nbytes %d\n" % affine_array[i].astype('float32').nbytes)
				o.write("affineshape %d %d\n" % (affine_array[i].shape[0], affine_array[i].shape[1]) )

		if num_object>0:
			for i in range(num_object):
				if not surfname==[]:
					o.write("surfname %s\n" % surfname[i])
				else:
					o.write("surfname unknown\n")

				o.write("element vertex\n")
				o.write("dtype float32\n")
				o.write("nbytes %d\n" % vertex_array[i].astype('float32').nbytes)
				o.write("vertexshape %d %d\n" % (vertex_array[i].shape[0], vertex_array[i].shape[1]) )

				o.write("element face\n")
				o.write("dtype uint32\n")
				o.write("nbytes %d\n" % face_array[i].astype('uint32').nbytes)
				o.write("faceshape %d %d\n" % (face_array[i].shape[0], face_array[i].shape[1]))

		if num_adjacency>0:
			for i in range(num_adjacency):
				o.write("element adjacency_object\n")
				o.write("dtype python_object\n")
				o.write("nbytes %d\n" % len(pickle.dumps(adjacency_array[i], -1)) )
				o.write("adjlength %d\n" % len(adjacency_array[i]) )

		if not subjectids==[]:
				o.write("element subject_id\n")
				o.write("dtype %s\n" % subjectids[0].dtype)
				o.write("nbytes %d\n" % subjectids[0].nbytes)
				o.write("listlength %d\n" % len(subjectids[0]))

		# create a recorded of what was added to the file. 'mode_add' denotes these items were added. tmi_history is expandable.
		tmi_history.append("history mode_add %d %d %d %d %d %d" % (currentTime, num_data, num_mask-h_mask, num_affine-h_affine, num_object-h_object, num_adjacency-h_adjacency) )
		for i in range(len(tmi_history)):
			o.write('%s\n' % (tmi_history[i]) )
		o.write("end_header\n")

		if output_binary:
			image_array = np.array(image_array.T, dtype='float32') # transpose to reduce file size
			image_array.tofile(o)
			if num_mask>0:
				for j in range(num_mask):
					binarymask = masking_array[j] * 1
					binarymask = np.array(binarymask.T, dtype=np.uint8)
					binarymask.tofile(o)
			if num_affine>0:
				for j in range(num_affine):
					outaffine = np.array(affine_array[j].T, dtype='float32')
					outaffine.tofile(o)
			if num_object>0:
				for j in range(num_object):
					outv = np.array(vertex_array[j].T, dtype='float32')
					outv.tofile(o)
					outf = np.array(face_array[j].T, dtype='uint32')
					outf.tofile(o)
			if not subjectids==[]:
				subjectids[0].tofile(o)
			if num_adjacency>0:
				for j in range(num_adjacency):
					pickle.dump(adjacency_array[j],o, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			if not image_array==[]:
				np.savetxt(o,image_array.astype(np.float32))
			if num_mask>0:
				for j in range(num_mask):
					binarymask = masking_array[j] * 1
					binarymask = np.array(binarymask, dtype=np.uint8)
					x, y, z = np.ma.nonzero(binarymask)
					for i in xrange(len(x)):
						o.write("%d %d %d\n" % (int(x[i]), int(y[i]), int(z[i]) ) )
			if num_affine>0:
				for j in range(num_affine):
					outaffine = np.array(affine_array[j])
					np.savetxt(o,outaffine.astype(np.float32))
			if not subjectids==[]:
				subjectids[0].tofile(o, sep='\n', format="%s")
			if num_object>0:
				for k in range(num_object):
					for i in xrange(len(vertex_array[k])):
						o.write("%1.6f %1.6f %1.6f\n" % (vertex_array[k][i,0], vertex_array[k][i,1], vertex_array[k][i,2] ) )
					for j in xrange(len(face_array[k])):
						o.write("%d %d %d\n" % (int(face_array[k][j,0]), int(face_array[k][j,1]), int(face_array[k][j,2]) ) )

		o.close()

###############
#  READ TMI   #
###############

def read_tm_filetype(tm_file, verbose=True):
	# getfilesize
	filesize = os.stat(tm_file).st_size
	# declare variables
	element = []
	element_dtype = []
	element_nbyte = []
	element_nmasked = []
	masking_array = []
	datashape = []
	maskshape = []
	maskname = []
	vertexshape = []
	faceshape = []
	surfname = []
	affineshape = []
	adjlength = []
	array_read = []
	object_read = []
	o_imgarray = []
	o_masking_array = []
	o_vertex = []
	o_face = []
	o_affine = []
	o_adjacency = []
	o_subjectids = []
	maskcounter = 0
	vertexcounter = 0
	facecounter = 0
	affinecounter = 0
	adjacencycounter = 0
	tmi_history = []

	# read first line
	obj = open(tm_file)
	reader = obj.readline().strip().split()
	firstword=reader[0]
	if firstword != 'tmi':
		print "Error: not a TFCE_mediation image."
		exit()
	reader = obj.readline().strip().split()
	firstword=reader[0]
	if firstword != 'format':
		print "Error: unknown reading file format"
		exit()
	else:
		tm_filetype = reader[1]


	while firstword != 'end_header':
		reader = obj.readline().strip().split()
		firstword=reader[0]
		if firstword == 'element':
			element.append((reader[1]))
		if firstword == 'dtype':
			element_dtype.append((reader[1]))
		if firstword == 'nbytes':
			element_nbyte.append((reader[1]))
		if firstword == 'datashape':
			datashape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword == 'nmasked':
			element_nmasked.append(( int(reader[1]) ))
		if firstword == 'maskshape':
			maskshape.append(np.array((reader[1], reader[2], reader[3])).astype(np.int))
		if firstword == 'maskname':
			maskname.append((reader[1]))
		if firstword == 'affineshape':
			affineshape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword == 'vertexshape':
			vertexshape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword == 'surfname':
			surfname.append((reader[1]))
		if firstword == 'faceshape':
			faceshape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword == 'adjlength':
			adjlength.append(np.array(reader[1]).astype(np.int))
		if firstword == 'history':
			tmi_history.append(str(' '.join(reader)))
		if firstword == 'listlength':
			listlength=int(reader[1])
	# skip header
	position = filesize
	for i in range(len(element_nbyte)):
		position-=int(element_nbyte[i])
	# readdata

	if tm_filetype == 'binary_little_endian':
		for e in range(len(element)):
			obj.seek(position)
			if verbose:
				print "reading %s" % str(element[e])
				print position
			if not str(element[e]) == 'adjacency_object':
				array_read.append((np.fromfile(obj, dtype=element_dtype[e])))
			else:
				object_read.append(pickle.load(obj))
			position += int(element_nbyte[e])
			# reshape arrays
			if str(element[e]) == 'data_array':
				o_imgarray.append(np.array(array_read[e][:datashape[0][0]*datashape[0][1]]).reshape(datashape[0][1],datashape[0][0]).T)
			if str(element[e]) == 'masking_array':
				masktemp = np.array(array_read[e][:(maskshape[maskcounter][2]*maskshape[maskcounter][1]*maskshape[maskcounter][0])]).reshape(maskshape[maskcounter][2],maskshape[maskcounter][1],maskshape[maskcounter][0]).T
				o_masking_array.append((np.array(masktemp, dtype=bool) ))
				maskcounter += 1
			if str(element[e]) == 'affine':
				o_affine.append(np.array(array_read[e][:affineshape[affinecounter][1]*affineshape[affinecounter][0]]).reshape(affineshape[affinecounter][1],affineshape[affinecounter][0]).T)
				affinecounter += 1
			if str(element[e]) == 'vertex':
				o_vertex.append(np.array(array_read[e][:vertexshape[vertexcounter][1]*vertexshape[vertexcounter][0]]).reshape(vertexshape[vertexcounter][1],vertexshape[vertexcounter][0]).T)
				vertexcounter += 1
			if str(element[e]) == 'face':
				o_face.append(np.array(array_read[e][:faceshape[facecounter][1]*faceshape[facecounter][0]]).reshape(faceshape[facecounter][1],faceshape[facecounter][0]).T)
				facecounter += 1
			if str(element[e]) == 'subject_id':
				o_subjectids.append((np.fromfile(array_read[e][:listlength], dtype=element_dtype[e])))
			if str(element[e]) == 'adjacency_object':
				o_adjacency.append(np.array(object_read[adjacencycounter][:adjlength[adjacencycounter]]))
				adjacencycounter += 1
	elif tm_filetype == 'ascii':
		for e in range(len(element)):
			if str(element[e]) == 'data_array':
				img_data = np.zeros((datashape[0][0], datashape[0][1]))
				for i in range(int(datashape[0][0])):
					img_data[i] = np.array(obj.readline().strip().split(), dtype = 'float32')
				o_imgarray.append((np.array(img_data, dtype = 'float32')))
			if str(element[e]) == 'masking_array':
					for k in range(element_nmasked[maskcounter]):
						masking_array.append((np.array(obj.readline().strip().split()).astype(np.int32)))
					masking_array = np.array(masking_array)
					outmask = np.zeros((maskshape[maskcounter]), dtype=np.int)
					outmask[masking_array[:,0],masking_array[:,1],masking_array[:,2]] = 1
					o_masking_array.append((np.array(outmask, dtype=bool)))
					maskcounter += 1
					masking_array=[]
			if str(element[e]) == 'affine':
				temparray = []
				for k in range(int(affineshape[affinecounter][0])):
					temparray.append((np.array(obj.readline().strip().split()).astype('float32')))
				o_affine.append((np.array(temparray, dtype='float32')))
				affinecounter += 1
			if str(element[e]) == 'vertex':
				temparray = []
				for k in range(int(vertexshape[vertexcounter][0])):
					temparray.append((np.array(obj.readline().strip().split()).astype('float32')))
				o_vertex.append((np.array(temparray, dtype='float32')))
				vertexcounter += 1
			if str(element[e]) == 'face':
				temparray = []
				for k in range(int(faceshape[facecounter][0])):
					temparray.append((np.array(obj.readline().strip().split()).astype('int32')))
				o_face.append(( np.array(temparray, dtype='int32') ))
				facecounter += 1
			if str(element[e]) == 'subject_id':
				temparray = []
				for k in range(listlength):
					temparray.append(obj.readline().strip() )
				o_subjectids.append(( np.array(temparray, dtype=element_dtype[e]) ))

	else:
		print "Error unknown filetype: %s" % tm_filetype
	return(element, o_imgarray, o_masking_array, maskname, o_affine, o_vertex, o_face, surfname, o_adjacency, tmi_history, o_subjectids) # add o_subjectids


###############
# CONVERT TMI #
###############

def convert_tmi(element, output_name, output_type='freesurfer', image_array=None, masking_array=None, affine_array=None, vertex_array=None, face_array=None):
	num_masks = 0
	num_affine = 0
	num_surf = 0
	for e in range(len(element)):
		if str(element[e]) == 'data_array':
			if image_array is not None:
				if masking_array is not None:
					num_masks = len(masking_array)
				if affine_array is not None:
					num_affine = len(affine_array)
		if str(element[e]) == 'vertex':
			if vertex_array is not None:
				if not len(vertex_array) == len(face_array):
					print "number of vertex and face elements must match"
					exit()
				num_surf = len(vertex_array)
	if output_type == 'freesurfer':
		if num_affine == 0:
			affine=None
		if image_array is not None:
			if num_masks == 1:
				savemgh_v2(image_array[0],masking_array[0], output_name, affine)
			elif len(masking_array) == 2:
				if affine is not None:
					affine  = affine_array[0]
				savemgh_v2(image_array[0][:len(masking_array[0][masking_array[0]==True])],masking_array[0], 'lh.%s' % output_name, affine)
				if affine is not None:
					affine  = affine_array[1]
				savemgh_v2(image_array[0][len(masking_array[0][masking_array[0]==True]):],masking_array[1], 'rh.%s' % output_name, affine)
			elif len(masking_array)>2:
				location = 0
				for i in range(len(masking_array)):
					if affine_array is not None:
						affine = affine_array[i]
					masklength=len(masking_array[i][masking_array[i]==True])
					savemgh_v2(image_array[0][location:(location+masklength)],masking_array[i], '%d.%s' % (i,output_name), affine)
					location+=masklength
		if num_surf == 1:
			save_fs(vertex_array[0], face_array[0], output_name)
		elif num_surf == 2:
			save_fs(vertex_array[0], face_array[0], 'lh.%s' % output_name)
			save_fs(vertex_array[1], face_array[1], 'rh.%s' % output_name)
		elif num_surf>2:
			for i in range(num_surf):
				save_fs(vertex_array[i], face_array[i], '%d.%s' % (i,output_name) )
	elif output_type == 'nifti':
		if num_affine == 0:
			affine=None
		else:
			affine=affine_array[0]
		if image_array is not None:
			savenifti_v2(image_array[0], masking_array[0], output_name, affine)
			savenifti_v2(np.ones(len(masking_array[0][masking_array[0]==True])), masking_array[0], 'mask.%s' % output_name, affine)
	else:
		print "Error. %s output type is not recognised" % output_type

