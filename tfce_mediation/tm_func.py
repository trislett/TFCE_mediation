#!/usr/bin/env python

#    Various functions for I/O functions for tm mmr
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
import math
import numpy as np
import nibabel as nib
from time import time
import matplotlib.pyplot as plt

from tfce_mediation.cynumstats import tval_int
from tfce_mediation.tm_io import savemgh_v2, savenifti_v2
from tfce_mediation.pyfunc import convert_redtoyellow, convert_bluetolightblue, convert_mpl_colormaps, calc_sobelz, convert_mni_object, convert_fs, convert_gifti, convert_ply

# Main Functions

# Mulitmodal Multisurface Regression
#
# Input:
# merge_y = the data array
# masking_array = the masking array
# pred_x = the predictor variable(s)
# calcTFCE = the TFCE function
# vdensity = the numper of each neighors at each point per unit distance
# position_array = the position of each mask in the image_array
# fullmask = concatenated mask of all masks
# perm_number = the permutation number
# randomise = randomisation flag
# verbose = longer output
# no_intercept = strip the intercept contrasts from the final results (default is true). Note, intercepts are always included in the regression model.
# set_surf_count = set the surface number for output
#
# Output:
# tvals = the t-value for all contrasts
# tfce_tvals = TFCE transformed values for postive associations
# neg_tfce_tvals = TFCE transformed values for negative associations
def calculate_tfce(merge_y, masking_array, pred_x, calcTFCE, vdensity, position_array, fullmask, perm_number = None, randomise = False, verbose = False, no_intercept = True, set_surf_count = None, print_interation = False):
	X = np.column_stack([np.ones(merge_y.shape[0]),pred_x])
	if randomise:
		np.random.seed(perm_number+int(float(str(time())[-6:])*100))
		X = X[np.random.permutation(list(range(merge_y.shape[0])))]
	k = len(X.T)
	invXX = np.linalg.inv(np.dot(X.T, X))
	tvals = tval_int(X, invXX, merge_y, merge_y.shape[0], k, merge_y.shape[1])
	if no_intercept:
		tvals = tvals[1:,:]
	tvals = tvals.astype(np.float32, order = "C")
	tfce_tvals = np.zeros_like(tvals).astype(np.float32, order = "C")
	neg_tfce_tvals = np.zeros_like(tvals).astype(np.float32, order = "C")

	for tstat_counter in range(tvals.shape[0]):
		tval_temp = np.zeros_like((fullmask)).astype(np.float32, order = "C")
		if tvals.shape[0] == 1:
			tval_temp[fullmask==1] = tvals[0]
		else:
			tval_temp[fullmask==1] = tvals[tstat_counter]
		tval_temp = tval_temp.astype(np.float32, order = "C")
		tfce_temp = np.zeros_like(tval_temp).astype(np.float32, order = "C")
		neg_tfce_temp = np.zeros_like(tval_temp).astype(np.float32, order = "C")
		calcTFCE.run(tval_temp, tfce_temp)
		calcTFCE.run((tval_temp*-1), neg_tfce_temp)
		tval_temp = tval_temp[fullmask==1]
		tfce_temp = tfce_temp[fullmask==1]
		neg_tfce_temp = neg_tfce_temp[fullmask==1]
#		position = 0
		for surf_count in range(len(masking_array)):
			start = position_array[surf_count]
			end = position_array[surf_count+1]
			if isinstance(vdensity, int): # check vdensity is a scalar
				tfce_tvals[tstat_counter,start:end] = (tfce_temp[start:end] * (tval_temp[start:end].max()/100) * vdensity)
				neg_tfce_tvals[tstat_counter,start:end] = (neg_tfce_temp[start:end] * ((tval_temp*-1)[start:end].max()/100) * vdensity)
			else:
				tfce_tvals[tstat_counter,start:end] = (tfce_temp[start:end] * (tval_temp[start:end].max()/100) * vdensity[start:end])
				neg_tfce_tvals[tstat_counter,start:end] = (neg_tfce_temp[start:end] * ((tval_temp*-1)[start:end].max()/100) * vdensity[start:end])
			if randomise:
				if set_surf_count is not None:
					os.system("echo %f >> perm_maxTFCE_surf%d_tcon%d.csv" % (np.nanmax(tfce_tvals[tstat_counter,start:end]),int(set_surf_count[surf_count]),tstat_counter+1))
					os.system("echo %f >> perm_maxTFCE_surf%d_tcon%d.csv" % (np.nanmax(neg_tfce_tvals[tstat_counter,start:end]),int(set_surf_count[surf_count]),tstat_counter+1))
				else:
					os.system("echo %f >> perm_maxTFCE_surf%d_tcon%d.csv" % (np.nanmax(tfce_tvals[tstat_counter,start:end]),surf_count,tstat_counter+1))
					os.system("echo %f >> perm_maxTFCE_surf%d_tcon%d.csv" % (np.nanmax(neg_tfce_tvals[tstat_counter,start:end]),surf_count,tstat_counter+1))
			else:
				if set_surf_count is not None:
					print("Maximum (untransformed) postive tfce value for surface %s, tcon %d: %f" % (int(set_surf_count[surf_count]),tstat_counter+1,np.nanmax(tfce_tvals[tstat_counter,start:end]))) 
					print("Maximum (untransformed) negative tfce value for surface %s, tcon %d: %f" % (int(set_surf_count[surf_count]),tstat_counter+1,np.nanmax(neg_tfce_tvals[tstat_counter,start:end])))
				else:
					print("Maximum (untransformed) postive tfce value for surface %s, tcon %d: %f" % (surf_count,tstat_counter+1,np.nanmax(tfce_tvals[tstat_counter,start:end]))) 
					print("Maximum (untransformed) negative tfce value for surface %s, tcon %d: %f" % (surf_count,tstat_counter+1,np.nanmax(neg_tfce_tvals[tstat_counter,start:end])))
		if verbose:
			print("T-contrast: %d" % tstat_counter)
			print("Max tfce from all surfaces = %f" % tfce_tvals[tstat_counter].max())
			print("Max negative tfce from all surfaces = %f" % neg_tfce_tvals[tstat_counter].max())
	if randomise:
		if print_interation:
			print("Interation number: %d" % perm_number)
#		os.system("echo %s >> perm_maxTFCE_allsurf.csv" % ( ','.join(["%0.2f" % i for i in tfce_tvals.max(axis=1)] )) )
#		os.system("echo %s >> perm_maxTFCE_allsurf.csv" % ( ','.join(["%0.2f" % i for i in neg_tfce_tvals.max(axis=1)] )) )
		tvals = None
		tfce_tvals = None
		neg_tfce_tvals = None
	tval_temp = None
	tfce_temp = None
	neg_tfce_temp = None
	del calcTFCE
	if not randomise:
		return (tvals.astype(np.float32, order = "C"), tfce_tvals.astype(np.float32, order = "C"), neg_tfce_tvals.astype(np.float32, order = "C"))


# Low Ram Mulitmodal Multisurface Regression
#
# Input:
# data = the data array
# mask = the masking array
# pred_x = the predictor variable(s)
# calcTFCE = the TFCE function
# vdensity = the numper of each neighors at each point per unit distance
# perm_number = the permutation number
# randomise = randomisation flag
# verbose = longer output
# no_intercept = strip the intercept contrasts from the final results (default is true). Note, intercepts are always included in the regression model.
# set_surf_count = set the surface number for output
#
# Output:
# tvals = the t-value for all contrasts
# tfce_tvals = TFCE transformed values for postive associations
# neg_tfce_tvals = TFCE transformed values for negative associations
def low_ram_calculate_tfce(data, mask, pred_x, calcTFCE, vdensity, set_surf_count = 0, perm_number = None, randomise = False, no_intercept = True, output_dir = None, perm_seed = None):
	X = np.column_stack([np.ones(data.shape[0]),pred_x])
	if randomise:
		if perm_seed is not None:
			np.random.seed(perm_number + perm_seed)
		else:
			np.random.seed(perm_number+int(float(str(time())[-6:])*100))
		X = X[np.random.permutation(list(range(data.shape[0])))]
	k = len(X.T)
	invXX = np.linalg.inv(np.dot(X.T, X))
	tvals = tval_int(X, invXX, data, data.shape[0], k, data.shape[1])
	if no_intercept:
		tvals = tvals[1:,:]
	tvals = tvals.astype(np.float32, order = "C")
	tfce_tvals = np.zeros_like(tvals).astype(np.float32, order = "C")
	neg_tfce_tvals = np.zeros_like(tvals).astype(np.float32, order = "C")
	for tstat_counter in range(tvals.shape[0]):
		tval_temp = np.zeros_like((mask)).astype(np.float32, order = "C")
		if tvals.shape[0] == 1:
			tval_temp[mask==1] = tvals[0]
		else:
			tval_temp[mask==1] = tvals[tstat_counter]

		tval_temp = tval_temp.astype(np.float32, order = "C")
		tfce_temp = np.zeros_like(tval_temp).astype(np.float32, order = "C")
		neg_tfce_temp = np.zeros_like(tval_temp).astype(np.float32, order = "C")
		calcTFCE.run(tval_temp, tfce_temp)
		calcTFCE.run(-tval_temp, neg_tfce_temp)

		tfce_tvals[tstat_counter,:] = (tfce_temp[mask==1] * (tval_temp.max()/100) * vdensity)
		neg_tfce_tvals[tstat_counter,:] = (neg_tfce_temp[mask==1] * ((tval_temp*-1).max()/100) * vdensity)

		if randomise:
			if output_dir is not None:
				permfile = "%s/perm_maxTFCE_surf%d_tcon%d.csv" % (output_dir, int(set_surf_count), tstat_counter+1)
			else:
				permfile = "perm_maxTFCE_surf%d_tcon%d.csv" % (int(set_surf_count), tstat_counter+1)
			os.system("echo %f >> %s" % (np.nanmax(tfce_tvals[tstat_counter,:]), permfile))
			os.system("echo %f >> %s" % (np.nanmax(neg_tfce_tvals[tstat_counter,:]), permfile))

	if not randomise:
		return (tvals.astype(np.float32, order = "C"), tfce_tvals.astype(np.float32, order = "C"), neg_tfce_tvals.astype(np.float32, order = "C"))


# Mulitmodal Multisurface Mediation
#
# Input:
# medtype = mediation type {I|M|Y}
# merge_y = the data array
# masking_array = the masking array
# pred_x = the upstream variable for mediation
# depend_y = the downstream variable for mediation
# calcTFCE = the TFCE function
# vdensity = the numper of each neighors at each point per unit distance
# position_array = the position of each mask in the image_array
# fullmask = concatenated mask of all masks
# perm_number = the permutation number
# randomise = randomisation flag
# verbose = longer output
#
# Output:
# SobelZ = the indirect effect statistic
# tfce_SobelZ = TFCE transformed indirect effect statistic
def calculate_mediation_tfce(medtype, merge_y, masking_array, pred_x, depend_y, calcTFCE, vdensity, position_array, fullmask, perm_number = None, randomise = False, verbose = False, no_intercept = True, print_interation = False):
	if randomise:
		np.random.seed(perm_number+int(float(str(time())[-6:])*100))
		indices_perm = np.random.permutation(list(range(merge_y.shape[0])))
		if (medtype == 'M') or (medtype == 'I'):
			if randomise:
				pred_x = pred_x[indices_perm]
		else:
			if randomise:
				pred_x = pred_x[indices_perm]
				depend_y = depend_y[indices_perm]
	SobelZ = calc_sobelz(medtype, pred_x, depend_y, merge_y, merge_y.shape[0], merge_y.shape[1])
	SobelZ = SobelZ.astype(np.float32, order = "C")
	tfce_SobelZ = np.zeros_like(SobelZ).astype(np.float32, order = "C")
	zval_temp = np.zeros_like((fullmask)).astype(np.float32, order = "C")
	zval_temp[fullmask==1] = SobelZ
	zval_temp = zval_temp.astype(np.float32, order = "C")
	tfce_temp = np.zeros_like(zval_temp).astype(np.float32, order = "C")
	calcTFCE.run(zval_temp, tfce_temp)
	zval_temp = zval_temp[fullmask==1]
	tfce_temp = tfce_temp[fullmask==1]
	for surf_count in range(len(masking_array)):
		start = position_array[surf_count]
		end = position_array[surf_count+1]
		if isinstance(vdensity, int): # check vdensity is a scalar
			tfce_SobelZ[start:end] = (tfce_temp[start:end] * (zval_temp[start:end].max()/100) * vdensity)
		else:
			tfce_SobelZ[start:end] = (tfce_temp[start:end] * (zval_temp[start:end].max()/100) * vdensity[start:end])
		if randomise:
			os.system("echo %f >> perm_maxTFCE_surf%d_%s_zstat.csv" % (np.nanmax(tfce_SobelZ[start:end]), surf_count, medtype))
		else:
			print("Max Sobel Z tfce value for surface %s:\t %1.5f" % (surf_count, np.nanmax(tfce_SobelZ[start:end]))) 
	if verbose:
		print("Max Zstat tfce from all surfaces = %f" % tfce_SobelZ.max())
	if randomise:
		print("Interation number: %d" % perm_number)
		SobelZ = None
		tfce_SobelZ = None
	zval_temp = None
	tfce_temp = None
	del calcTFCE
	if not randomise:
		return (SobelZ.astype(np.float32, order = "C"), tfce_SobelZ.astype(np.float32, order = "C"))

# Low Ram Mulitmodal Multisurface Mediation
#
# Input:
# medtype = mediation type {I|M|Y}
# data = the data array
# mask = the masking array
# pred_x = the predictor variable(s)
# calcTFCE = the TFCE function
# vdensity = the numper of each neighors at each point per unit distance
# perm_number = the permutation number
# randomise = randomisation flag
# verbose = longer output
# no_intercept = strip the intercept contrasts from the final results (default is true). Note, intercepts are always included in the regression model.
# set_surf_count = set the surface number for output
#
# Output:
# SobelZ = the indirect effect statistic
# tfce_SobelZ = TFCE transformed indirect effect statistic
def low_ram_calculate_mediation_tfce(medtype, data, mask, pred_x, depend_y, calcTFCE, vdensity, set_surf_count = 0, perm_number = None, randomise = False, no_intercept = True, output_dir = None, perm_seed = None):

	if randomise:
		if perm_seed is not None:
			np.random.seed(perm_number + perm_seed)
		else:
			np.random.seed(perm_number+int(float(str(time())[-6:])*100))
		indices_perm = np.random.permutation(list(range(data.shape[0])))

		if (medtype == 'M') or (medtype == 'I'):
			if randomise:
				pred_x = pred_x[indices_perm]
		else:
			if randomise:
				pred_x = pred_x[indices_perm]
				depend_y = depend_y[indices_perm]

	SobelZ = calc_sobelz(medtype, pred_x, depend_y, data, data.shape[0], data.shape[1])
	SobelZ = SobelZ.astype(np.float32, order = "C")
	tfce_SobelZ = np.zeros_like(SobelZ).astype(np.float32, order = "C")
	zval = np.zeros_like((mask)).astype(np.float32, order = "C")
	zval[mask==1] = SobelZ
	zval = zval.astype(np.float32, order = "C")
	tfce_zval = np.zeros_like(zval).astype(np.float32, order = "C")
	calcTFCE.run(zval, tfce_zval)
	zval = zval[mask==1]
	tfce_zval = tfce_zval[mask==1]
	tfce_zval = (tfce_zval * (zval.max()/100) * vdensity)

	if randomise:
		if output_dir is not None:
			permfile = "%s/perm_maxTFCE_surf%d_%s_zstat.csv" % (output_dir, set_surf_count, medtype)
		else:
			permfile = "perm_maxTFCE_surf%d_%s_zstat.csv" % (set_surf_count, medtype)
		os.system("echo %f >> %s" % (np.nanmax(tfce_zval), permfile))
	else:
		return (zval.astype(np.float32, order = "C"), tfce_zval.astype(np.float32, order = "C"))


# Mulitmodal Multisurface Regression for mixed TFCE setting (E, H)
#
# Input:
# assigntfcesettings = the designation of the H and E setting to used for each mask
# merge_y = the data array
# masking_array = the masking array
# position_array = the position of each mask in the image_array
# vdensity = the numper of each neighors at each point per unit distance
# pred_x = the predictor variables
# calcTFCE = the TFCE functions with different settings
# perm_number = the permutation number
# randomise = randomisation flag
# medtype = mediation type {I|M|Y}
# depend_y = the downstream variable for mediation
#
# Output:
# tvals = the t-value for all contrasts
# tfce_tvals = TFCE transformed values for postive associations
# neg_tfce_tvals = TFCE transformed values for negative associations
def calc_mixed_tfce(assigntfcesettings, merge_y, masking_array, position_array, vdensity, pred_x, calcTFCE, perm_number = None, randomise = False, medtype = None, depend_y = None):
	for i in np.unique(assigntfcesettings):
		data_mask = np.zeros(merge_y.shape[1], dtype=bool)
		extract_range = np.argwhere(assigntfcesettings==i)
		for surface in extract_range:
			start = position_array[int(surface)]
			end = position_array[int(surface)+1]
			data_mask[start:end] = True
		subset_merge_y = merge_y[:,data_mask]
		try:
			temp_vdensity = vdensity[data_mask]
		except:
			temp_vdensity = 1

		if randomise:
			
			if i == 0:
				display_iter = True
			else:
				display_iter = False

			calculate_tfce(subset_merge_y, 
				np.array(masking_array)[assigntfcesettings==i], 
				pred_x, 
				calcTFCE[i], 
				temp_vdensity, 
				create_position_array(np.array(masking_array)[assigntfcesettings==i]),
				create_full_mask(np.array(masking_array)[assigntfcesettings==i]),
				set_surf_count = extract_range,
				perm_number=perm_number, 
				randomise = True,
				print_interation = display_iter)
		else:
			temp_tvals, temp_tfce_tvals, temp_neg_tfce_tvals = calculate_tfce(subset_merge_y, 
				np.array(masking_array)[assigntfcesettings==i], 
				pred_x, 
				calcTFCE[i], 
				temp_vdensity, 
				create_position_array(np.array(masking_array)[assigntfcesettings==i]),
				create_full_mask(np.array(masking_array)[assigntfcesettings==i]),
				set_surf_count = extract_range)

			if i == 0: # at first instance
				tvals = tfce_tvals = neg_tfce_tvals = np.zeros((temp_tvals.shape[0],merge_y.shape[1]))
			tvals[:,data_mask] = temp_tvals
			tfce_tvals[:,data_mask] = temp_tfce_tvals
			neg_tfce_tvals[:,data_mask] = temp_neg_tfce_tvals
			temp_tvals = None
			temp_tfce_tvals = None
			temp_neg_tfce_tvals = None
	if not randomise:
		return tvals, tfce_tvals, neg_tfce_tvals
#		if not medtype:
#			return tvals, tfce_tvals, neg_tfce_tvals
#		else:
#			return SobelZ, tfce_SobelZ


# Apply FWER correction to TFCE transformed stastics to output corrected 1-p-value image that are either FWER(study) or FWER(modality)
#
# Input:
# image_array = the data array
# num_contrasts = the number constrasts
# surface_range = across which masks to perform pFWER corrtion
# num_perm = the number of permuation performed
# num_surf = the number of surfaces maskes
# tminame = the *.tmi file name
# position_array = the position of each mask in the image_array
# pos_range = which TFCE transformed statistics images are from positive direction associations
# neg_range = which TFCE transformed statistics images are from negative direction associations
# method = default is 'scale' which transforms the null distribution of each permuted maximum TFCE value into the same space for all masks
# weight = weight the scaled maximum null distribution based on voxel/vertex size of the mask
#
# Output:
# positive_data = corrected 1-p-value images from positive assocations
# negative_data = corrected 1-p-value images from negative assocations
def apply_mfwer(image_array, num_contrasts, surface_range, num_perm, num_surf, tminame, position_array, pos_range, neg_range = None, method = 'scale', weight = None, mediation = False, medtype = None): 
	# weight = None is essentially Tippet without considering mask size

	maxvalue_array = np.zeros((num_perm,num_contrasts))
	temp_max = np.zeros((num_perm, num_surf))
	positive_data = np.zeros((image_array[0].shape[0],num_contrasts))
	if mediation == False:
		negative_data = np.zeros((image_array[0].shape[0],num_contrasts))

	if weight == 'logmasksize':
		x = []
		for i in range(len(position_array)-1):
			x.append((position_array[i+1] - position_array[i]))
		weights = (np.log(x)/np.log(x).sum())/np.mean(np.log(x)/np.log(x).sum()) # weights between mask size and max values probablity.
		w_temp_max = np.zeros((num_perm, num_surf))

	for contrast in range(num_contrasts):
		for surface in surface_range: # the standardization is done within each surface
			if mediation == False:
				log_perm_results = np.log(np.genfromtxt('output_%s/perm_maxTFCE_surf%d_tcon%d.csv' % (tminame,surface,contrast+1))[:num_perm])
			else:
				log_perm_results = np.log(np.genfromtxt('output_%s/perm_maxTFCE_surf%d_%s_zstat.csv' % (tminame,surface,str(medtype)))[:num_perm])
			# set log(0) back to zero (only happens with small masks and very large effect sizes)
			log_perm_results[np.isinf(log_perm_results)] = 0
			start = position_array[surface]
			end = position_array[surface+1]

			# log transform, standarization
			posvmask = np.log(image_array[0][start:end,pos_range[contrast]]) > 0
			temp_lt = np.log(image_array[0][start:end,pos_range[contrast]][posvmask]) # log and z transform the images by the permutation values (the max tfce values are left skewed)
			temp_lt -= log_perm_results.mean()
			temp_lt /= log_perm_results.std()
			temp_lt += 10
			positive_data[start:end,contrast][posvmask] = temp_lt
			del temp_lt, posvmask

			if mediation == False:
				posvmask = np.log(image_array[0][start:end,neg_range[contrast]]) > 0
				temp_lt = np.log(image_array[0][start:end,neg_range[contrast]][posvmask]) # log and z transform the images by the permutation values (the max tfce values are left skewed)
				temp_lt -= log_perm_results.mean()
				temp_lt /= log_perm_results.std()
				temp_lt += 10
				negative_data[start:end,contrast][posvmask] = temp_lt
				del temp_lt, posvmask

			log_perm_results -= log_perm_results.mean() # standardize the max TFCE values 
			log_perm_results /= log_perm_results.std()
			if weight == 'logmasksize':
				w_log_perm_results = log_perm_results * weights[surface]
				w_log_perm_results += 10
				w_temp_max[:,surface] = w_log_perm_results
			log_perm_results += 10
			temp_max[:,surface] = log_perm_results


		if not weight == None:
			w_temp_max[np.isnan(w_temp_max)]=0
			max_index = np.argmax(w_temp_max, axis=1)
			max_value_list = np.zeros((len(max_index)),dtype=np.float32) # this should not be necessary
			for i in range(len(temp_max)):
				max_value_list[i]= temp_max[i,max_index[i]]
			maxvalue_array[:,contrast] = np.sort(max_value_list)
			del max_value_list
		else:
			maxvalue_array[:,contrast] = np.sort(temp_max.max(axis=1))
	del temp_max
	# two for loops just so my brain doesn't explode
	for contrast in range(num_contrasts):
		sorted_perm_tfce_max=maxvalue_array[:,contrast]
		p_array=np.zeros_like(sorted_perm_tfce_max)
		corrp_img = np.zeros((positive_data.shape[0]))
		for j in range(num_perm):
			p_array[j] = np.true_divide(j,num_perm)
		cV=0
		for k in positive_data[:,contrast]:
			corrp_img[cV] = find_nearest(sorted_perm_tfce_max,k,p_array)
			cV+=1
		positive_data[:,contrast] = np.copy(corrp_img)
		if mediation == False:
			cV=0
			corrp_img = np.zeros((negative_data.shape[0]))
			for k in negative_data[:,contrast]:
				corrp_img[cV] = find_nearest(sorted_perm_tfce_max,k,p_array)
				cV+=1
			negative_data[:,contrast] = np.copy(corrp_img)
	if mediation == False:
		return positive_data, negative_data
	else:
		return positive_data

# Additional Functions

# create a mega mask of all of the mask included in a *tmi
#
# Input: 
# masking_array from tmi file
#
# Output:
# full mask of all surfacees
def create_full_mask(masking_array):
	# make mega mask
	fullmask = []
	for i in range(len(masking_array)):
		if masking_array[i].shape[2] == 1: # check if vertex or voxel image
			fullmask = np.hstack((fullmask, masking_array[i][:,0,0]))
		else:
			fullmask = np.hstack((fullmask, masking_array[i][masking_array[i]==True]))
	return fullmask


# Merge adjacency sets and makes all pointers unique
#
# Input: 
# adjacent_range = which adjacency sets to merge
# adjacency_array = adjacency sets included in the tmi file
#
# Output:
# adjacency = adjacency for all masks in tmi file
def merge_adjacency_array(adjacent_range, adjacency_array):
	v_count = 0
	if len(adjacent_range) == 1:
		adjacency = np.copy(adjacency_array[0])
		for i in range(len(adjacency)):
			adjacency[i] = np.array(list(adjacency[i])).tolist() # list fixes 'set' 'int' error
	else:
		for e in adjacent_range:
			if v_count == 0:
				adjacency = np.copy(adjacency_array[0])
				for i in range(len(adjacency)):
					adjacency[i] = np.array(list(adjacency[i])).tolist() # list fixes 'set' 'int' error
				v_count += len(adjacency_array[e])
			else:
				temp_adjacency = np.copy(adjacency_array[e])
				for i in range(len(adjacency_array[e])):
					temp_adjacency[i] = np.add(list(temp_adjacency[i]), v_count).tolist() # list fixes 'set' 'int' error
				adjacency = np.hstack((adjacency, temp_adjacency))
				v_count += len(adjacency_array[e])
	return adjacency


# checks the permutation files and make sure that they are all the same length
def lowest_length(num_contrasts, surface_range, tmifilename, medtype = None):
	lengths = []
	for contrast in range(num_contrasts):
		for surface in surface_range: # the standardization is done within each surface
			if medtype is not None:
				lengths.append(np.array(np.genfromtxt('output_%s/perm_maxTFCE_surf%d_%s_zstat.csv' % (tmifilename,surface,medtype)).shape[0]))
			else:
				lengths.append(np.array(np.genfromtxt('output_%s/perm_maxTFCE_surf%d_tcon%d.csv' % (tmifilename,surface,contrast+1)).shape[0]))
	return np.array(lengths).min()


# Marks the data range for all masks in the data_array of a tmi file
def create_position_array(masking_array):
	pointer = 0
	position_array = [0]
	for i in range(len(masking_array)):
		pointer += len(masking_array[i][masking_array[i]==True])
		position_array.append(pointer)
	return position_array


#find nearest permuted TFCE max value that corresponse to family-wise error rate 
def find_nearest(array, value, p_array):
	idx = np.searchsorted(array, value, side="left")
	if idx == len(p_array):
		return p_array[idx-1]
	elif math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
		return p_array[idx-1]
	else:
		return p_array[idx]


# paints the threshold values on a surface
def paint_surface(lowthresh, highthres, color_scheme, data_array, save_colorbar = True):
	colormaps = np.array(plt.colormaps(),dtype=np.str)
	if (str(color_scheme) == 'r_y') or (str(color_scheme) == 'red-yellow'):
		out_color_array = convert_redtoyellow(np.array((float(lowthresh),float(highthres))), data_array, save_colorbar = save_colorbar)
	elif (str(color_scheme) == 'b_lb') or (str(color_scheme) == 'blue-lightblue'):
		out_color_array = convert_bluetolightblue(np.array((float(lowthresh),float(highthres))), data_array, save_colorbar = save_colorbar)
	elif np.any(colormaps == str(color_scheme)):
		out_color_array = convert_mpl_colormaps(np.array((float(lowthresh),float(highthres))), data_array, str(color_scheme), save_colorbar = save_colorbar)
	else:
		print("Error: colour scheme %s does not exist" % str(color_scheme))
		quit()
	return out_color_array


# strips basename
def strip_basename(basename):
	if basename.endswith('.mgh'):
		basename = basename[:-4]
	elif basename.endswith('.nii.gz'):
		basename = basename[:-7]
	else:
		pass
	if basename.startswith('lh.all'):
		basename = 'lh.%s' % basename[7:]
	if basename.startswith('rh.all'):
		basename = 'rh.%s' % basename[7:]
	return basename


# chooses neuroimaging file type to save based on data shape
def saveauto(image_array, index, imagename, affine=None):
	if index.shape[2] > 1:
		savenifti_v2(image_array, index, imagename, affine)
	else:
		savemgh_v2(image_array, index, imagename, affine)


# replaces a mask file and prompts for replacement name
def replacemask(orig_mask, orig_maskname, maskfile):
	mask = nib.load(maskfile)
	mask_data = mask.get_data()
	_, mask_index = maskdata(mask_data)
	if not orig_mask.shape == mask_index.shape:
		print("Error. Replacement mask dimensions %s does not match the original mask dimensions %s" % (mask_index.shape, orig_mask.shape))
		sys.exit()
	splash_warning = input("Warning replacing the mask will irreversibly change the masked data array. Enter Yes to confirm: \n")
	if splash_warning == 'Yes':
		pass
	elif splash_warning == 'Y':
		pass
	elif splash_warning == 'yes':
		pass
	elif splash_warning == 'y':
		pass
	else:
		print("Replace mask canceled. Exiting.")
		quit()
	newname = input("Enter a new mask name or press enter to keep the exist replacement name: %s\n" % orig_maskname)
	if newname:
		mask_name = newname
	else:
		mask_name = orig_maskname
	return (mask_index, mask_name)


# replaces a surface file
def replacesurface(orig_v, orig_f, surf_filename):
	surf_ext = os.path.splitext(surf_filename)[1]
	surfname = os.path.basename(surf_filename)
	if surf_ext == '.ply':
		v, f = convert_ply(surf_filename)
	elif surf_ext == '.gii':
		v, f = convert_gifti(surf_filename)
	elif surf_ext == '.obj':
		print("Reading surface as MNI object")
		v, f = convert_mni_object(surf_filename)
	elif surf_ext == '.srf':
		v, f = convert_fs(surf_filename)
	else:
		v, f = convert_fs(surf_filename) # place holder
	if not orig_v.shape == v.shape:
		print("Error. Vertices shape mismatch.")
		sys.exit()
	if not orig_f.shape == f.shape:
		print("Warning. Faces shape mismatch.")
	return  v, f, surfname


# quick function to correctly mask data
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

# Prints history and basic info from a tmi file
#
# Input:
#
# tmi_history = history from tmi file
# maskname_array = mask names from tmi file
# surfname = surface names from tmi file
# (optional) num_con = the number of contrasts (i.e., image_array[0].shape[1])
#
# Output:
# None
def print_tmi_history(tmi_history, maskname_array, surfname, num_con = None, contrast_names = []):
	num_masks = 0
	num_affines = 0
	num_surfaces = 0
	num_adjac = 0
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
		print("# masks: \t %s" % line[4])
		print("# affines: \t %s" % line[5])
		print("# surfaces: \t %s" % line[6])
		print("# adjacency sets: \t %s\n" % line[7])

	print("--- Mask names ---")
	for i in range(len(maskname_array)):
		print("Mask %d : %s" % (i,maskname_array[i]))
	print("")
	print("--- Surface names ---")
	for i in range(len(surfname)):
		print("Surface %d : %s" % (i,surfname[i]))
	print("")
	if contrast_names != []:
		print("--- Contrasts/Subjects ---")
		count = 0
		for contrast in contrast_names[0]:
			print("Contrast %d : %s" % (count,contrast))
			count += 1
		print("")
		num_con = None
	if num_con:
		print("--- Contrasts/Subjects ---")
		print(list(range(num_con)))
		print("")
	print("--- Total ---")
	print("# masks: \t %d \t ([0 -> %d])" % (num_masks, num_masks-1))
	print("# affines: \t %d \t ([0 -> %d])" % (num_affines, num_affines-1))
	print("# surfaces: \t %d \t ([0 -> %d])" % (num_surfaces, num_surfaces-1))
	print("# adjacency sets: \t %d \t ([0 -> %d])\n" % (num_adjac, num_adjac-1))


