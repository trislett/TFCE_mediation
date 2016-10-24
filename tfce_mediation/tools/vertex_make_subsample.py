#!/usr/bin/python

import sys
import numpy as np
import nibabel as nib


if len(sys.argv) < 3:
	print "*******************************************************************"
	print "Usage: %s [1D Subgrouping Variable] [surface (area or thickness)]" % (str(sys.argv[0]))
	print "  "
	print "Careful. Creates a subgroub based on missing data from an analysis."
	print "The subgrouping variable should be a 1D text file list with missing"
	print "variables coded as a string (e.g. NA or NaN)."
	print "--- You must be in same directory as python_temp ---"
	print "*******************************************************************"
else:
	cmdargs = str(sys.argv)
	arg_subgroupvariable = str(sys.argv[1])
	surftype = str(sys.argv[2])
	if len(sys.argv)==3:
		fwhm = str('03B')
	else:
		fwhm = str(sys.argv[3])
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
	outdata_mask_lh = data_full_lh[:,:,:,1]
	subdata_lh=data_lh[:,masking_variable]


	data_full_rh = img_surf_rh.get_data()
	data_rh = np.squeeze(data_full_rh)
	affine_mask_rh = img_surf_rh.get_affine()
	outdata_mask_rh = data_full_rh[:,:,:,1]
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
