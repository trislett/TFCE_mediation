#!/usr/bin/env python

import numpy as np
import nibabel as nib
import argparse as ap

DESCRIPTION = "Creates a subgroub based on missing data from an analysis. The subgrouping variable should be a 1D text file list with missing variables coded as a string (e.g. NA or NaN)."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	ap.add_argument("-i", "--input", 
		help="1D Subgrouping Variable] [surface (area or thickness)]", 
		nargs=2, 
		metavar=('*.csv','surface'),
		required=True)
	ap.add_argument("-f", "--fwhm", 
		help="FWHM of all surface file (Default: %(default)s))",
		nargs=1,
		default=['03B'],
		metavar=('??B'))
	return ap

def run(opts):
	arg_subgroupvariable = str(opts.input[0])
	surftype = str(opts.input[1])
	fwhm = str(opts.fwhm[0])
	img_surf_lh = nib.freesurfer.mghformat.load('lh.all.%s.%s.mgh' % (surftype,fwhm))
	img_surf_rh = nib.freesurfer.mghformat.load('rh.all.%s.%s.mgh' % (surftype,fwhm))
	subgroupvariable = np.genfromtxt(arg_subgroupvariable, delimiter=',')
	masking_variable=np.isfinite(subgroupvariable)
	data_full_lh = img_surf_lh.get_data()
	data_lh = np.squeeze(data_full_lh)
	if data_lh.shape[1] > len(subgroupvariable):
		print "Error. Number of subjects doesn't equal length of subgrouping variable"
		exit()
	affine_mask_lh = img_surf_lh.get_affine()
	subdata_lh=data_lh[:,masking_variable]

	data_full_rh = img_surf_rh.get_data()
	data_rh = np.squeeze(data_full_rh)
	affine_mask_rh = img_surf_rh.get_affine()
	subdata_rh=data_rh[:,masking_variable]

	num_voxel = subdata_lh.shape[0]
	num_subjects = subdata_lh.shape[1]
	out_subdata_lh = np.zeros([num_voxel, 1, 1, num_subjects])
	out_subdata_rh= np.zeros([num_voxel, 1, 1, num_subjects])
	nib.save(nib.Nifti1Image(data_full_lh,affine_mask_lh),'lh.all.%s.%s.backup.mgh' % (surftype,fwhm))
	nib.save(nib.Nifti1Image(data_full_rh,affine_mask_rh),'rh.all.%s.%s.backup.mgh' % (surftype,fwhm))
	out_subdata_lh[:,0,0,:] = subdata_lh
	out_subdata_rh[:,0,0,:] = subdata_rh
	nib.save(nib.Nifti1Image(out_subdata_lh,affine_mask_lh),'lh.all.%s.%s.mgh' % (surftype,fwhm))
	nib.save(nib.Nifti1Image(out_subdata_rh,affine_mask_rh),'rh.all.%s.%s.mgh' % (surftype,fwhm))
	np.savetxt('masking_variable.out', masking_variable, delimiter=',',fmt='%i')

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
