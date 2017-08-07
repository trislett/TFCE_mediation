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

from __future__ import division
import os
import sys
import numpy as np
import nibabel as nib
import argparse as ap
from time import gmtime, strftime

from tfce_mediation.pyfunc import convert_mni_object, convert_fs, convert_gifti, convert_ply
from tfce_mediation.tm_io import write_tm_filetype, read_tm_filetype
from tfce_mediation.pyfunc import zscaler

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
		print "Error: %d dimensions are not supported." % data.ndim
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
	ap.add_argument("-r", "--revert",
		help="Revert tmi to earlier time-point (removed elements cannot be restored!). Make sure to check the history first (-oh) or by using tm_multimodal read-tmi-header. Input the time-point that you wish to revert the tmi file. e.g. -r 5",
		nargs=1, 
		metavar='int',
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
	element, image_array, masking_array, maskname_array, affine_array, vertex_array, face_array, surfname, adjacency_array, tmi_history, subjectids = read_tm_filetype(opts.inputtmi[0])

	num_masks = 0
	num_affines = 0
	num_surfaces = 0
	num_adjac = 0
	append_history = True

	if opts.outputnewtmi:
		outname = opts.outputnewtmi[0]
	else:
		outname = opts.inputtmi[0]

	if opts.history:
		print "--- History ---"
		for i in range(len(tmi_history)):
			print "Time-point %d" % i
			line = tmi_history[i].split(' ')
			print ("Date: %s-%s-%s %s:%s:%s" % (line[2][6:8],line[2][4:6],line[2][0:4], line[2][8:10], line[2][10:12], line[2][12:14]) )
			if line[1]=='mode_add':
				print "Elements added"
				num_masks += int(line[4])
				num_affines += int(line[5])
				num_surfaces += int(line[6])
				num_adjac += int(line[7])
			elif line[1]=='mode_sub':
				print "Elements removed"
				num_masks -= int(line[4])
				num_affines -= int(line[5])
				num_surfaces -= int(line[6])
				num_adjac -= int(line[7])
			else:
				print "Error: mode is not understood"
			print "Number of masks: %s" % line[4]
			print "Number of affines: %s" % line[5]
			print "Number of surfaces: %s" % line[6]
			print "Number of adjacency sets: %s\n" % line[7]

		print "--- Total ---"
		print "Number of masks: %d ([0 -> %d])" % (num_masks, num_masks-1)
		print "Number of affines: %d ([0 -> %d])" % (num_affines, num_affines-1)
		print "Number of surfaces: %d ([0 -> %d])" % (num_surfaces, num_affines-1)
		print "Number of adjacency sets: %d ([0 -> %d])\n" % (num_adjac, num_affines-1)
		quit()
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
	# Write tmi file
	if not image_array==[]:
		write_tm_filetype(outname, output_binary = opts.outputtype=='binary', image_array=np.vstack(image_array), masking_array=masking_array, maskname=maskname,  affine_array=affine_array, vertex_array=vertex_array, face_array=face_array, surfname=surfname, adjacency_array=adjacency_array, checkname=False, tmi_history=tmi_history, append_history=append_history)
	else:
		write_tm_filetype(outname, output_binary = opts.outputtype=='binary', masking_array=masking_array, maskname=maskname,  affine_array=affine_array, vertex_array=vertex_array, face_array=face_array, surfname=surfname, adjacency_array=adjacency_array, checkname=False, tmi_history=tmi_history, append_history=append_history)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
