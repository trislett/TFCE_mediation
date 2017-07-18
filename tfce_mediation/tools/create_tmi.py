#!/usr/bin/env python

#    Build *.tmi images for TFCE_mediation
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

DESCRIPTION = "Build a tmi file."

#arguments parser
def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	#input type

	outname = ap.add_mutually_exclusive_group(required=True)
	outname.add_argument("-o", "--outputname",
		help="Output file name.",
		nargs=1, 
		metavar='*.tmi')
	outname.add_argument("-a", "--append",
		help="Append information to an existing *.tmi file.",
		nargs=1, 
		metavar='*.tmi')

	ap.add_argument("-i", "--inputimages",
		help="Input neuroimage(s) in *nii, *mgh or *mnc. Note, this should be used for group data (i.e., 4D images).",
		nargs='+', 
		metavar=('MGH or NIFTI or MNC'))
	ap.add_argument("-c_i", "--concatenateimages",
		help="Input a set of neuroimages that are then concatenated. The images should be in the same space. Only one set of concatenated can be added at a time. To add multiple modalities (or surfaces) used the rerun %(prog)s --append (the number of subjects has to be the same). (e.g. -c_i FAtoStd/Subject*_FAtoStd.nii.gz",
		nargs='+', 
		metavar=('MGH or NIFTI or MNC'))
	ap.add_argument("-s", "--scale",
		help="Scale each image that is being concatenated. Must be used with -c_i. Useful when importing timeseries data.",
		action = 'store_true')
	ap.add_argument("--concatenatename",
		help="Specify a 4D image name for concatenated images. Must be used with -c_i.",
		nargs=1)

	ap.add_argument("-i_text", "--inputtext",
		help="Input neuroimage(s) in *nii, *mgh or *mnc. Note, this should be used for group data (i.e., 4D images).",
		nargs='+', 
		metavar=('*.txt or *.csv'))
	ap.add_argument("-c_text", "--concatenatetext",
		help="Input text files with a single column that will be concatenated. The text should be in the same space (i.e., same length). Only one set of concatenated can be added at a time. To add multiple modalities used the rerun %(prog)s --append (the number of subjects has to be the same).",
		nargs='+', 
		metavar=('*.txt or *.csv'))
	ap.add_argument("-c_bin", "--concatenatebinary",
		help="Input a set of binary images that encoded <f (float) that are then concatenated. The input should be check afterwards. Each image should be of the same length. To add multiple modalities (or surfaces) used the rerun %(prog)s --append. (e.g. [for ENIGMA-shape output], -c_bin Subject*/LogJacs_10.raw)",
		nargs='+', 
		metavar=('*.bin or *.raw'))

	ap.add_argument("-i_masks", "--inputmasks",
		help="Input masks (recommended). There should be same number of input images as masks. The file formats do not have to match, but the masks should be in binary format. If masks are not entered, they will be created using non-zero data.",
		nargs='+', 
		metavar=('MGH or NIFTI or MNC'))

	in_surface = ap.add_mutually_exclusive_group(required=False)
	in_surface.add_argument("-i_fs", "--inputfreesurfer",
		help="Input a freesurfer surface (e.g., -i_fs $SUBJECTS_DIR/fsaverage/surf/lh.midthickness $SUBJECTS_DIR/fsaverage/surf/rh.midthickness)", 
		nargs='+', 
		metavar=('*'))
	in_surface.add_argument("-i_gifti", "--inputgifti",
		help="Input a gifti surface file (e.g., --i_gifti average.surf.gii)", 
		nargs='+', 
		metavar=('*.surf.gii'))
	in_surface.add_argument("-i_mni", "--inputmniobj",
		help="Input a MNI object file (e.g., --i_mni l_hemi.obj r_hemi.obj)", 
		nargs='+', 
		metavar=('*.obj'))
	in_surface.add_argument("-i_ply", "--inputply",
		help="Input a MNI object file (e.g., --i_ply l_hemi.ply). Note, vertex colors will be stripped.", 
		nargs='+', 
		metavar=('*.ply'))
	ap.add_argument("-i_adj", "--inputadjacencyobject",
		help="Input adjacency objects for each surface. There should be same number of input images as masks.",
		nargs='+', 
		metavar=('*.npy'))

	ap.add_argument("--outputtype",
		default='binary',
		const='binary',
		nargs='?',
		choices=['binary','ascii'],
		help="Set output type. (default: %(default)s). The ascii option will not store adjacency objects.")

	return ap

def run(opts):
	image_array = []
	affine_array = []
	masking_array = []
	maskname = []
	vertex_array=[]
	face_array=[]
	surfname = []
	adjacency_array=[]
	tmi_history=[]

	if opts.outputname:
		outname = opts.outputname[0]
		if not outname.endswith('tmi'):
			if opts.outputtype=='binary':
				if not outname.endswith('tmi'):
					outname += '.tmi'
			else:
				outname += '.ascii.tmi'
	if opts.append:
		outname = opts.append[0]
		_, image_array, masking_array, maskname, affine_array, vertex_array, face_array, surfname, adjacency_array, tmi_history, subjectids = read_tm_filetype(outname)

	if opts.inputimages:
		for i in range(len(opts.inputimages)):
			basename, file_ext = os.path.splitext(opts.inputimages[i])
			if file_ext == '.gz':

				os.system("zcat %s > %s" % (opts.inputimages[i],basename))
				img = nib.load('%s' % basename)
			else:
				img = nib.load(opts.inputimages[i])
			img_data = img.get_data()


			if opts.inputmasks:
				mask = nib.load(opts.inputmasks[i])
				mask_data = mask.get_data()
				if not np.array_equal(img_data.shape[:3], mask_data.shape[:3]):
					print "Error mask data dimension do not fit image dimension"
					exit()
				mask_data = mask_data==1
				img_data = img_data[mask_data]
			else:
				img_data, mask_data = maskdata(img_data)
			masking_array.append(np.array(mask_data))
			image_array.append(np.array(img_data))
			affine_array.append(img.affine)
			maskname.append(np.array(os.path.basename(opts.inputimages[i])))
			if file_ext == '.gz':
				os.system("rm %s" % basename)

	if opts.concatenateimages:
		img = nib.load(opts.concatenateimages[0])
		img_data = img.get_data()
		numMerge=len(opts.concatenateimages)

		if opts.inputmasks:
			if not len(opts.inputmasks)==1:
				print "Only one mask can be added using concatenate. See help (hint: rerun using append option for multiple modalities/surfaces)"
				exit()
			mask = nib.load(opts.inputmasks[0])
			mask_data = mask.get_data()
			if not np.array_equal(img_data.shape[:3], mask_data.shape[:3]):
				print "Error mask data dimension do not fit image dimension"
				exit()
			mask_data = mask_data==1
			img_data = img_data[mask_data].astype(np.float32)
		else:
			print "Creating mask from first image."
			img_data, mask_data = maskdata(img_data)

		for i in xrange(numMerge):
			print "merging image %s" % opts.concatenateimages[i]
			if i > 0:
				tempdata = nib.load(opts.concatenateimages[i]).get_data()
				tempdata = tempdata[mask_data].astype(np.float32)
				if opts.scale: 
					tempdata = zscaler(tempdata.T).T
				img_data = np.column_stack((img_data,tempdata))
			else:
				if opts.scale: 
					img_data = zscaler(img_data.T).T

		masking_array.append(np.array(mask_data))
		image_array.append(np.array(img_data))
		affine_array.append(img.affine)
		if opts.concatenatename:
			maskname.append(np.array(os.path.basename(opts.concatenatename[0])))

	if opts.inputtext:
		for i in range(len(opts.inputtext)):
			#img_data = np.genfromtxt(opts.inputtext[i], delimiter=',') # slower, more ram usage
			img_data = []
			with open('lh_cortical_thickness_for_all_subjects.csv') as data_file:
				for line in data_file:
					img_data.append(line.strip().split(','))
			img_data = np.array(img_data).astype(np.float32)

			img_data, mask_data = maskdata(img_data)
			masking_array.append(np.array(mask_data))
			image_array.append(np.array(img_data))

	if opts.concatenatetext:

		firstimg_data = np.genfromtxt(opts.concatenatetext[0], delimiter=',')
		numMerge=len(opts.concatenatetext)

		for i in xrange(numMerge):
			print "merging text file %s" % opts.concatenatetext[i]
			if i > 0:
				tempdata = np.genfromtxt(opts.concatenatetext[i], delimiter=',')
			img_data = np.column_stack((img_data,tempdata))
		img_data, mask_data = maskdata(img_data)
		masking_array.append(np.array(mask_data))
		image_array.append(np.array(img_data))

	if opts.concatenatebinary:

		firstimg_data = np.fromfile(opts.concatenatebinary[0], dtype = 'f')
		numMerge=len(opts.concatenatebinary)

		for i in xrange(numMerge):
			print "merging simple float binary file %s" % opts.concatenatebinary[i]
			if i > 0:
				tempdata = np.fromfile(opts.concatenatebinary[i], dtype = 'f')
			img_data = np.column_stack((img_data,tempdata))
		img_data, mask_data = maskdata(img_data)
		masking_array.append(np.array(mask_data))
		image_array.append(np.array(img_data))

	if opts.inputfreesurfer:
		for i in range(len(opts.inputfreesurfer)):
			v,f = convert_fs(str(opts.inputfreesurfer[i]))
			vertex_array.append(v)
			face_array.append(f)
			surfname.append(np.array(os.path.basename(opts.inputfreesurfer[i])))
	if opts.inputgifti:
		for i in range(len(opts.inputgifti)):
			v,f = convert_gifti(str(opts.inputgifti[i]))
			vertex_array.append(v)
			face_array.append(f)
			surfname.append(np.array(os.path.basename(opts.inputgifti[i])))
	if opts.inputmniobj:
		for i in range(len(opts.inputmniobj)):
			v,f = convert_mni_object(str(opts.inputmniobj[i]))
			vertex_array.append(v)
			face_array.append(f)
			surfname.append(np.array(os.path.basename(opts.inputmniobj[i])))
	if opts.inputply:
		for i in range(len(opts.inputply)):
			v,f = convert_ply(str(opts.inputply[i]))
			vertex_array.append(v)
			face_array.append(f)
			surfname.append(np.array(os.path.basename(opts.inputply[i])))
	if opts.inputadjacencyobject:
		for i in range(len(opts.inputadjacencyobject)):
			adjacency_array.append(np.load(str(opts.inputadjacencyobject[i])))
		if not np.equal(len(adjacency_array),len(masking_array)):
			if not len(adjacency_array) % len(masking_array) == 0:
				print "Number of adjacency objects does not match number of images."
			else:
				print "Error number of adjacency objects is not divisable by the number of masking arrays."
				exit()

	# Write tmi file
	if not image_array==[]:
		write_tm_filetype(outname, output_binary = opts.outputtype=='binary', image_array=np.vstack(image_array), masking_array=masking_array, maskname=maskname,  affine_array=affine_array, vertex_array=vertex_array, face_array=face_array, surfname=surfname, adjacency_array=adjacency_array, checkname=False, tmi_history=tmi_history)
	else:
		write_tm_filetype(outname, output_binary = opts.outputtype=='binary', masking_array=masking_array, maskname=maskname,  affine_array=affine_array, vertex_array=vertex_array, face_array=face_array, surfname=surfname, adjacency_array=adjacency_array, checkname=False, tmi_history=tmi_history)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
