#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import argparse as ap

def loadnifti(imagename):
	if os.path.exists(imagename): # check if file exists
		if imagename.endswith('.nii.gz'):
			os.system("zcat %s > temp.nii" % imagename)
			img = nib.load('temp.nii')
			img_data = img.get_data()
			os.system("rm temp.nii")
		else:
			img = nib.load(imagename)
			img_data = img.get_data()
	else:
		print "Cannot find input image: %s" % imagename
		exit()
	return (img,img_data)

def loadmgh(imagename):
	if os.path.exists(imagename): # check if file exists
		img = nib.freesurfer.mghformat.load(imagename)
		img_data = img.get_data()
	else:
		print "Cannot find input image: %s" % imagename
		exit()
	return (img,img_data)

def savenifti(imgdata, img, index, imagename):
	outdata = imgdata.astype(np.float32, order = "C")
	if imgdata.ndim == 2:
		imgout = np.zeros((img.shape[0],img.shape[1],img.shape[2],outdata.shape[1]))
	elif imgdata.ndim == 1:
		imgout = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
	else:
		print 'error'
	imgout[index]=outdata
	nib.save(nib.Nifti1Image(imgout.astype(np.float32, order = "C"),img.affine),imagename)

def savemgh(imgdata, img, index, imagename):
	outdata = imgdata.astype(np.float32, order = "C")
	if imgdata.ndim == 2:
		imgout = np.zeros((img.shape[0],img.shape[1],img.shape[2],outdata.shape[1]))
	elif imgdata.ndim == 1:
		imgout = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
	else:
		print 'error'
	imgout[index]=outdata
	nib.save(nib.freesurfer.mghformat.MGHImage(imgout.astype(np.float32, order = "C"),img.affine),imagename)

DESCRIPTION = "Fast merging for Nifti or MGH images"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	datatype = ap.add_mutually_exclusive_group(required=True)
	datatype.add_argument("--voxel", 
		help="Voxel input",
		action="store_true")
	datatype.add_argument("--vertex", 
		help="Vertex input",
		action="store_true")
	ap.add_argument("-o", "--output", nargs=1, help="[4D_image]", metavar=('*.nii.gz'), required=True)
	ap.add_argument("-i", "--input", nargs='+', help="[3Dimage] ...", metavar=('*.nii.gz'), required=True)
	ap.add_argument("-m", "--mask", nargs=1, help="[3Dimage]", metavar=('*.nii.gz'))
	return ap

def run(opts):
	if opts.voxel:
		img, img_data = loadnifti(opts.input[0])
	if opts.vertex:
		img, img_data = loadmgh(opts.input[0])
	numMerge=len(opts.input)
	outname=opts.output[0]

	if opts.mask:
		if opts.voxel:
			mask , data_mask = loadnifti(opts.mask[0])
		if opts.vertex:
			mask , data_mask = loadmgh(opts.mask[0])
		mask_index = data_mask>0.99
	else:
		mask_index = np.zeros((img_data.shape[0],img_data.shape[1],img_data.shape[2]))
		mask_index = (mask_index == 0)
	img_data_trunc = img_data[mask_index]

	for i in xrange(numMerge):
		print "merging image %s" % opts.input[i]
		if i > 0:
			if opts.voxel:
				_, tempimgdata = loadnifti(opts.input[i])
			if opts.vertex:
				_, tempimgdata = loadmgh(opts.input[i])

			tempimgdata=tempimgdata[mask_index]
			img_data_trunc = np.column_stack((img_data_trunc,tempimgdata))
	if opts.voxel:
		savenifti(img_data_trunc, img, mask_index, outname)
	if opts.vertex:
		savemgh(img_data_trunc, img, mask_index, outname)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)




