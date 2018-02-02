#!/usr/bin/env python

#    Edit *.tmi images for TFCE_mediation
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


import os
import sys
import numpy as np
import nibabel as nib
import argparse as ap
from time import gmtime, strftime

from tfce_mediation.tm_io import write_tm_filetype, read_tm_filetype
from tfce_mediation.tm_func import replacemask, replacesurface

def maskdata(data):
	if data.ndim==4:
		mean = data.mean(axis=3)
		mask = mean!=0
		data = data[mask]
	elif data.ndim==3:
		mask = data!=0
		data = data[mask]
	elif data.ndim==2:
		mask = np.zeros((data.shape[0],1,1))
		mean = data.mean(axis=1)
		mask[mean!=0,0,0]=1
		mask = mask==1
		data = data[mean!=0]
	elif data.ndim==1: #build 3D mask
		mask = np.zeros((data.shape[0],1,1))
		mask[data!=0,0,0]=1
		mask = mask==1
		data = data[data!=0]
	else:
		print("Error: %d dimensions are not supported." % data.ndim)
		exit()
	return (data, mask)

DESCRIPTION = "Edit tmi file."

#arguments parser
def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):

	ap.add_argument("-i_tmi", "--inputtmi",
		help="Edit existing *.tmi file.",
		nargs=1, 
		metavar='*.tmi',
		required=True)
	ap.add_argument("-oh", "--history",
		help="Output tmi file history and exits.", 
		action='store_true')
	ap.add_argument("-os", "--outputstats",
		help="Output min/max values from value for each contrast per mask and exits.", 
		action='store_true')
	ap.add_argument("--revert",
		help="Revert tmi to earlier time-point (removed elements cannot be restored!). Make sure to check the history first (-oh) or by using tm_multimodal read-tmi-header. Input the time-point that you wish to revert the tmi file. e.g. -r 5",
		nargs=1, 
		metavar='int',
		required=False)
	ap.add_argument("-rm", "--replacemask",
		help="Edit existing *.tmi file.",
		nargs=2, 
		metavar=('int', 'nii.gz|minc|mgh'),
		required=False)
	ap.add_argument("-rom", "--reordermasks",
		help="Edit existing *.tmi file.",
		nargs='+', 
		metavar=('int'),
		type=int,
		required=False)
	ap.add_argument("-ra", "--replaceaffine",
		help="Replace existing affine with inputted one. Must specifiy affine, and input 4x4 affine (as a *.csv). e.g., -ra 2 new_affine.csv",
		nargs=2, 
		metavar=('int', '*.csv'),
		required=False)
	ap.add_argument("-roa", "--reorderaffines",
		help="Reorder the existing affines. The number of inputs must match the number of affines. e.g., -rom 3 4 1 2",
		nargs='+', 
		metavar=('int'),
		type=int,
		required=False)
	ap.add_argument("-rs", "--replacesurface",
		help="Replace existing surface with inputted one. Must specifiy surface to replace, and input a freesurface object (no extension or *.srf). e.g., -rs 2 lh.midthickness.",
		nargs=2, 
		metavar=('int', '*.srf'),
		required=False)
	ap.add_argument("-ros", "--reordersurfaces",
		help="Reorder the existing surfaces. The number of inputs must match the number of surfaces. e.g., -ros 3 4 1 2. Note, both vertices and faces will be reordered.",
		nargs='+',
		type=int,
		metavar=('int'),
		required=False)
	ap.add_argument("-radj", "--replaceadj",
		help="Replace existing adjacency with inputted one. Must specifiy adjacency, and input new adjacency (as a *.npy). e.g., -radj 2 lh.midthickness.adj.3mm.npy",
		nargs=2,
		metavar=('int', '*.npy'),
		required=False)
	ap.add_argument("-roadj", "--reorderadj",
		help="Reorder the existing adjacency sets. The number of inputs must match the number of adjacency sets. e.g., -roadj 3 4 1 2.",
		nargs='+',
		type=int,
		metavar=('int'),
		required=False)
	ap.add_argument("-d", "--delete",
		help="Remove element(s). Input the type {mask|affine|surface|adjacency} and the element number or range. e.g. -d mask 2 3 OR -d surface 3",
		nargs='+',
		metavar=('element','int'),
		required=False)
	ap.add_argument("-o", "--outputnewtmi",
		help="Output a new tmi file (instead of editing existing one).",
		nargs=1,
		metavar='*.tmi',
		required=False)

	return ap

def run(opts):
	currentTime=int(strftime("%Y%m%d%H%M%S",gmtime()))
	# read tmi
	element, image_array, masking_array, maskname_array, affine_array, vertex_array, face_array, surfname, adjacency_array, tmi_history, columnids = read_tm_filetype(opts.inputtmi[0])

	num_masks = 0
	num_affines = 0
	num_surfaces = 0
	num_adjac = 0
	append_history = True

	print(len(sys.argv))
	
	# if not enough inputs, output history
	if len(sys.argv) <= 4:
		opts.history = True

	if opts.outputnewtmi:
		outname = opts.outputnewtmi[0]
	else:
		outname = opts.inputtmi[0]


	# first, index data array
	pointer = 0
	position_array = [0]
	for i in range(len(masking_array)):
		pointer += len(masking_array[i][masking_array[i]==True])
		position_array.append(pointer)
	del pointer

	if opts.outputstats:
		for i in range(len(columnids[0])):
			print("\n --- Subject/Contrast[%d]: %s ---\n"  % (i, columnids[0][i]))
			for j, m in enumerate(maskname_array):
				start = position_array[j]
				end = position_array[j+1]
				print("Mask[%d]\t%s \t [%1.4f, %1.4f]" % (j, m,
					image_array[0][start:end,i].min(),
					image_array[0][start:end,i].max()))
		if surfname is not None:
			print("\n --- Surfaces ---\n")
			for s, surf in enumerate(surfname):
				print("Surface[%d]\t%s" % (s,surf)) 
		sys.exit()

	if opts.history:
		print("--- History ---")
		for i in range(len(tmi_history)):
			print("Time-point %d" % i)
			line = tmi_history[i].split(' ')
			print(("Date: %s-%s-%s %s:%s:%s" % (line[2][6:8],line[2][4:6],line[2][0:4], line[2][8:10], line[2][10:12], line[2][12:14]) ))
			if line[1]=='mode_add':
				print("Elements added")
				num_masks += int(line[4])
				num_affines += int(line[5])
				num_surfaces += int(line[6])
				num_adjac += int(line[7])
			elif line[1]=='mode_sub':
				print("Elements removed")
				num_masks -= int(line[4])
				num_affines -= int(line[5])
				num_surfaces -= int(line[6])
				num_adjac -= int(line[7])
			elif line[1] == 'mode_replace':
				print("Element replaced")
			elif line[1] == 'mode_reorder':
				print("Element reordered")
			else:
				print("Error: mode is not understood")
			print("# masks: %s" % line[4])
			print("# affines: %s" % line[5])
			print("# surfaces: %s" % line[6])
			print("# adjacency sets: %s\n" % line[7])

		print("--- Mask names ---")
		for i in range(len(maskname_array)):
			print("Mask %d : %s" % (i,maskname_array[i]))
		print("")
		print("--- Surface names ---")
		for i in range(len(surfname)):
			print("Surface %d : %s" % (i,surfname[i]))
		print("")
		print("--- Total ---")
		print("# masks: %d ([0 -> %d])" % (num_masks, num_masks-1))
		print("# affines: %d ([0 -> %d])" % (num_affines, num_affines-1))
		print("# surfaces: %d ([0 -> %d])" % (num_surfaces, num_surfaces-1))
		print("# adjacency sets: %d ([0 -> %d])\n" % (num_adjac, num_adjac-1))
		quit()
	# revert
	if opts.revert:
		for i in range(int(opts.revert[0])+1):
#		for i in range(int(6)):
			line = tmi_history[i].split(' ')
			if line[1]=='mode_add':
				num_masks += int(line[4])
				num_affines += int(line[5])
				num_surfaces += int(line[6])
				num_adjac += int(line[7])
			elif line[1]=='mode_sub':
				num_masks -= int(line[4])
				num_affines -= int(line[5])
				num_surfaces -= int(line[6])
				num_adjac -= int(line[7])
		size_data = 0
		for i in range(num_masks):
			size_data += len(masking_array[i][masking_array[i]==True])
		image_array[0] = image_array[0][:size_data,:]
		masking_array = masking_array[:num_masks]
		maskname_array = maskname_array[:num_masks]
		affine_array = affine_array[:num_affines]
		vertex_array = vertex_array[:num_surfaces]
		face_array = face_array[:num_surfaces]
		surfname = surfname[:num_surfaces]
		adjacency_array = adjacency_array[:num_adjac]

		# edit history

		orig_num_masks = 0
		orig_num_affines = 0
		orig_num_surfaces = 0
		orig_num_adjac = 0
		for i in range(len(tmi_history)):
			line = tmi_history[i].split(' ')
			if line[1]=='mode_add':
				orig_num_masks += int(line[4])
				orig_num_affines += int(line[5])
				orig_num_surfaces += int(line[6])
				orig_num_adjac += int(line[7])
			elif line[1]=='mode_sub':
				orig_num_masks -= int(line[4])
				orig_num_affines -= int(line[5])
				orig_num_surfaces -= int(line[6])
				orig_num_adjac -= int(line[7])
		tmi_history.append("history mode_sub %d %d %d %d %d %d" % (currentTime, 1, orig_num_masks-num_masks, orig_num_affines-num_affines, orig_num_surfaces - num_surfaces, orig_num_adjac-num_adjac))
		append_history = False

	# masks
	if opts.replacemask:

		original_mask = np.copy(masking_array[int(opts.replacemask[0])])
		print("Replacing mask %s" % maskname_array[int(opts.replacemask[0])])
		masking_array[int(opts.replacemask[0])], maskname_array[int(opts.replacemask[0])] = replacemask(original_mask, maskname_array[int(opts.replacemask[0])], opts.replacemask[1])

		size_oldmask = original_mask[original_mask==True].shape[0]
		size_newmask = masking_array[int(opts.replacemask[0])][masking_array[int(opts.replacemask[0])]==True].shape[0]
		diff_masks = size_oldmask - size_newmask
		new_data_array = np.zeros((int(image_array[0].shape[0] - diff_masks), image_array[0].shape[1]))

		pointer = 0
		for surface in range(len(masking_array)):
			start = position_array[surface]
			end = position_array[surface+1]
			size = end - start + pointer # CHECK THIS !
			if int(surface) == int(opts.replacemask[0]):
				size = size_newmask + pointer
				tempdata = np.zeros((original_mask.shape[0], original_mask.shape[1], original_mask.shape[2],image_array[0].shape[1]))
				tempdata[original_mask] = image_array[0][start:end,:]
				new_data_array[pointer:size,:] = tempdata[masking_array[surface]]
			else:
				new_data_array[pointer:size,:] = image_array[0][start:end,:]
			pointer += size # CHECK THIS !
		image_array[0] = new_data_array
		del new_data_array

		tmi_history.append("history mode_replace %s 1 1 0 0 0" % currentTime)
		append_history = False
	if opts.reordermasks:
		if not len(opts.reordermasks) == len(masking_array):
			print("Error. The number of inputs [%d] must match the number of masks [%] in %s." % (len(opts.reordermasks),len(masking_array),opts.inputtmi[0]))
			sys.exit()
		print("Reordering \n%s\n to \n%s\n" % (str(maskname_array),str([maskname_array[i] for i in opts.reordermasks])))
		masking_array = [masking_array[i] for i in opts.reordermasks]
		maskname_array = [maskname_array[i] for i in opts.reordermasks]
		tmi_history.append("history mode_reorder %d 1 %d 0 0 0" % (currentTime, len(opts.reordermasks)))
		append_history = False

		# get surface coordinates in data array
		pointer = 0
		position_array = [0]
		for i in range(len(masking_array)):
			pointer += len(masking_array[i][masking_array[i]==True])
			position_array.append(pointer)
		del pointer
		new_data_array = np.zeros_like(image_array[0])
		pointer = 0
		for surface in opts.reordermasks:
			start = position_array[surface]
			end = position_array[surface+1]
			size = end - start + pointer # CHECK THIS !
			new_data_array[pointer:size,:] = image_array[0][start:end,:]
			pointer += size # CHECK THIS !
		image_array[0] = new_data_array
		del new_data_array

	# affines
	if opts.replaceaffine:
		original_affine = affine_array[int(opts.replaceaffine[0])]
		new_affine = np.genfromtxt(opts.replaceaffine[1], delimiter=',')
		if np.isnan(new_affine).any():
			new_affine = np.genfromtxt(opts.replaceaffine[1], delimiter=' ')
		if np.isnan(new_affine).any():
			new_affine = np.genfromtxt(opts.replaceaffine[1], delimiter='\t')
		if np.isnan(new_affine).any():
			print("Error. Input file must be a comma, space, or tab separate (if they are, check that the line ending is UNIX and utf-8)).")
			sys.exit()
		if not new_affine.shape == (4, 4):
			print("Error. Input affine must (4,4).")
			sys.exit()
		print("Replacing affine:\n %s\n" % str(original_affine))
		print("With affine:\n %s\n" % str(new_affine))
		affine_array[int(opts.replaceaffine[0])] = new_affine
		tmi_history.append("history mode_replace %d 1 0 1 0 0" % currentTime)
		append_history = False
	if opts.reorderaffines:
		if not len(opts.reorderaffines) == len(affine_array):
			print("Error. The number of inputs [%d] must match the number of affines [%] in %s." % (len(opts.reorderaffines),len(affine_array),opts.inputtmi[0]))
			sys.exit()
		print("Reordering %s to %s" % (str(list(range(len(affine_array)))),str(opts.reorderaffines)))
		affine_array = [affine_array[i] for i in opts.reorderaffines]
		maskname_array = [maskname_array[i] for i in opts.reorderaffines]
		tmi_history.append("history mode_reorder %d 1 %d 0 0 0" % (currentTime,len(opts.reorderaffines)))
		append_history = False

	# surfaces
	if opts.replacesurface:
		orig_v = vertex_array[int(opts.replacesurface[0])]
		orig_f = face_array[int(opts.replacesurface[0])]
		vertex_array[int(opts.replacesurface[0])], face_array[int(opts.replacesurface[0])], surfname[int(opts.replacesurface[0])] = replacesurface(orig_v, orig_f, opts.replacesurface[1])
		tmi_history.append("history mode_replace %d 1 0 0 1 0" % currentTime)
		append_history = False
	if opts.reordersurfaces:
		if not len(opts.reordersurfaces) == len(vertex_array):
			print("Error. The number of inputs [%d] must match the number of surfaces [%] in %s." % (len(opts.reordersurfaces),len(vertex_array),opts.inputtmi[0]))
			sys.exit()
		print("Reordering \n%s\n to \n%s\n" % (str(surfname),str([surfname[i] for i in opts.reordersurfaces])))
		vertex_array = [vertex_array[i] for i in opts.reordersurfaces]
		face_array = [face_array[i] for i in opts.reordersurfaces]
		surfname = [surfname[i] for i in opts.reordersurfaces]
		tmi_history.append("history mode_reorder %d 1 0 0 %d 0" % (currentTime,len(opts.reordersurfaces)))
		append_history = False

	# adjacency sets
	if opts.replaceadj:
		original_adj = adjacency_array[int(opts.replaceadj[0])]
		new_adj = np.load(opts.replaceadj[1])

		if not len(original_adj) == len(new_adj):
			print("Error. Adjacency set lengths must match.")
			sys.exit()
		affine_array[int(opts.replaceadj[0])] = new_adj
		tmi_history.append("history mode_replace %d 1 0 0 0 1" % currentTime)
		append_history = False
	if opts.reorderadj:
		if not len(opts.reorderadj) == len(adjacency_array):
			print("Error. The number of inputs [%d] must match the number of adjacency sets [%] in %s." % (len(opts.reorderadj),len(affine_array),opts.inputtmi[0]))
			sys.exit()
		print("Reordering %s to %s" % (str(list(range(len(adjacency_array)))),str(opts.reorderadj)))
		adjacency_array = [adjacency_array[i] for i in opts.reorderadj]
		tmi_history.append("history mode_reorder %d 1 0 0 0 %d" % (currentTime,len(opts.reorderadj)))
		append_history = False

	# delete element
	if opts.delete:
		if len(opts.delete) == 2:
			delete_range = np.array([int(opts.delete[1])])
		elif len(opts.delete) == 3:
			delete_range = list(range(int(opts.delete[1]), (int(opts.delete[2])+1)))
		else:
			print(opts.delete)
			print(len(opts.delete))
			print("Error. -d option can only be a single value or a range")
			sys.exit()
		if opts.delete[0] == 'mask':
			arr_size = len(masking_array)

			mask_mask = np.ones(len(masking_array), dtype=bool)
			data_mask = np.ones(len(image_array[0]), dtype=bool)

			pointer = 0
			position_array = [0]
			for i in range(len(masking_array)):
				pointer += len(masking_array[i][masking_array[i]==True])
				position_array.append(pointer)
			del pointer
			new_data_array = np.zeros_like(image_array[0])

			for surface in delete_range:
				start = position_array[surface]
				end = position_array[surface+1]
				data_mask[start:end] = False
			image_array[0] = image_array[0][data_mask]

			mask_mask[delete_range] = False
			masking_array = np.array(masking_array)[mask_mask]
			maskname_array = np.array(maskname_array)[mask_mask]

			tmi_history.append("history mode_sub %d 1 %d 0 0 0" % (currentTime,int(len(delete_range))))
			append_history = False

		elif opts.delete[0] == 'affine':
			arr_size = len(affine_array)
			mask = np.ones(len(affine_array), dtype=bool)
			mask[delete_range] = False
			affine_array = np.array(affine_array)[mask]
			tmi_history.append("history mode_sub %d 1 0 %d 0 0" % (currentTime,int(len(delete_range))))
			append_history = False
		elif opts.delete[0] == 'surface':
			arr_size = len(vertex_array)
			mask = np.ones(len(vertex_array), dtype=bool)
			mask[delete_range] = False
			vertex_array = np.array(vertex_array)[mask]
			face_array = np.array(face_array)[mask]
			surfname = np.array(surfname)[mask]
			tmi_history.append("history mode_sub %d 1 0 0 %d 0" % (currentTime,int(len(delete_range))))
			append_history = False
		elif opts.delete[0] == 'adjacency':
			arr_size = len(adjacency_array)
			mask = np.ones(len(adjacency_array), dtype=bool)
			mask[delete_range] = False
			adjacency_array = np.array(adjacency_array)[mask]
			tmi_history.append("history mode_sub %d 1 0 0 0 %d" % (currentTime,int(len(delete_range))))
			append_history = False
		else: 
			print("Error. Type must be one of the following: {mask|affine|surface|adjacency}")
			sys.exit()

	# Write tmi file
	if not image_array==[]:
		write_tm_filetype(outname, 
			output_binary = 'binary',
			image_array=np.vstack(image_array),
			masking_array=masking_array,
			maskname=maskname_array,
			affine_array=affine_array,
			vertex_array=vertex_array,
			face_array=face_array,
			surfname=surfname,
			adjacency_array=adjacency_array,
			checkname=False,
			columnids=np.array(columnids),
			tmi_history=tmi_history,
			append_history=append_history)
	else:
		write_tm_filetype(outname,
			output_binary = 'binary',
			masking_array=masking_array,
			maskname=maskname_array,
			affine_array=affine_array,
			vertex_array=vertex_array,
			face_array=face_array,
			surfname=surfname,
			adjacency_array=adjacency_array,
			checkname=False,
			columnids=np.array(columnids),
			tmi_history=tmi_history,
			append_history=append_history)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
