#!/usr/bin/python

import os
import numpy as np
import nibabel as nib
import argparse as ap

ap = ap.ArgumentParser(description="fast split of 4D nifti image")

ap.add_argument("-i", "--input", nargs=1, help="[4D_image]", metavar=('*.nii.gz'), required=True)
#ap.add_argument("-o", "--output", nargs=1, help="[outname]", metavar=('vol_????.nii.gz'), default=('vol'))
opts = ap.parse_args()

os.system("zcat %s > temp_4d.nii" % opts.input[0])
img_all = nib.load('temp_4d.nii')
data_all = img_all.get_data()
affine_all = img_all.get_affine()
header_all = img_all.get_header()
numImages=data_all.shape[3]
#outhead=str(opts.output[0])

for i in range(int(numImages)):
	nib.save(nib.Nifti1Image(data_all[:,:,:,i],affine_all),"vol%05d.nii.gz" % (i))
os.system("rm temp_4d.nii")
