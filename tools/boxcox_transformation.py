#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
from scipy import stats
from joblib import Parallel, delayed

np.seterr(all='ignore')

def transform_boxcox(data_array,out_data_array,i):
	out_data_array,_= stats.boxcox(data_array)
	return out_data_array

def boxcox(data_array, num_vertex, num_cores):
	bc_data = np.zeros(data_short.shape)
	bc_data = Parallel(n_jobs=num_cores)(delayed(transform_boxcox)((data_short[i,:]+1),bc_data[i,:],i) for i in xrange(num_vertex))
	print "Transformation finished"
	return np.array(bc_data)

#def slow_boxcox(data_array, num_vertex, num_cores = 1):
#	bc_data = np.zeros(data_array.shape)
#	for i in xrange(num_vertex):
#		print i
#		bc_data[i,:],_=stats.boxcox((data_array[i,:]+1))
#	return bc_data

if len(sys.argv) < 3:
	print "*******************************************************************"
	print "Usage: %s [surface file *.mgh] [# of processors]" % (str(sys.argv[0]))
	print "  "
	print "Apply Box-Cox Transformation to a 4D file, and apply 3mm smoothing"
	print "Box-Cox Transformation uses parallel processing"
	print "Example: %s lh.all.area.00.mgh 12" % (str(sys.argv[0]))
	print "*******************************************************************"
else:
	cmdargs = str(sys.argv)
	surface = str(sys.argv[1])
	num_cores = int(sys.argv[2])
	surf_name = surface.split('.mgh',1)[0]
	surf_gen = surface.split('.00.',1)[0]
	hemi = surface.split('.',1)[0]
	if not os.path.exists('python_temp_area'):
		os.mkdir('python_temp_area')
	img_surf = nib.freesurfer.mghformat.load(surface)
	data_full = img_surf.get_data()
	data_short = np.squeeze(data_full)
	affine_mask = img_surf.get_affine()
	num_vertex = data_short.shape[0]
	num_subjects = data_short.shape[1]
	outdata_mask = data_full[:,:,:,1]
	low_values_indices = data_short < 0
	data_short[low_values_indices] = 0
	bc_data = boxcox(data_short, num_vertex, num_cores)
	np.save('python_temp_area/bc_data',bc_data)
	out_bc_data=data_full
	out_bc_data[:,0,0,:] = bc_data
	nib.save(nib.freesurfer.mghformat.MGHImage(out_bc_data,affine_mask),"%s.boxcox.mgh" % surf_name)
	print "Smoothing %s.boxcox.mgh" %  surf_name 
	os.system = os.popen("mri_surf2surf --hemi %s --s fsaverage --sval %s.boxcox.mgh --fwhm 3 --cortex --tval %s.03B.boxcox.mgh" % (hemi, surf_name, surf_gen))


