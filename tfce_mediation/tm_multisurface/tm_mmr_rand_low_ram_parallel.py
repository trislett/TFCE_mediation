#!/usr/bin/env python

#    TFCE_mediation TMI multimodality, multisurface multiple regression
#    Copyright (C) 2017  Tristram Lett

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
import numpy as np
import argparse as ap
from time import time

from tfce_mediation.cynumstats import resid_covars
from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.tm_io import read_tm_filetype, write_tm_filetype, savemgh_v2, savenifti_v2
from tfce_mediation.pyfunc import save_ply, convert_voxel, vectorized_surface_smooth
from tfce_mediation.tm_func import calculate_tfce, calculate_mediation_tfce, calc_mixed_tfce, apply_mfwer, create_full_mask, merge_adjacency_array, lowest_length, create_position_array, paint_surface, strip_basename, saveauto, low_ram_calculate_tfce, low_ram_calculate_mediation_tfce

DESCRIPTION = "Companion program for mmr-lr"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):

	ap.add_argument("-sn", "--surfacenumber",
		nargs=1, 
		type=int,
		metavar=('INT'),
		required=False)
	ap.add_argument("-pr", "--permutationrange",
		nargs=2, 
		type=int,
		metavar=('INT','INT'),
		required=False)
	ap.add_argument("-os", "--outputstats",  
		help = "Calculates the stats without permutation testing, and outputs the tmi.", 
		action = 'store_true')
	ap.add_argument("--path",
		nargs=1, 
		type=str,
		metavar=('STR'),
		required=True)
	ap.add_argument("--seed",
		nargs=1, 
		type=int,
		metavar=('INT'))
	return ap


def run(opts):
	try:
		sopts = np.load("tmi_temp/opts.npy").tolist()
	except:
		print("Error: tmi_temp was not found. Run mmr-lr first, or change directory to where tmi_temp is located.") # this error should never happen.

	tfce_settings = []
	masking_array = np.load("tmi_temp/masking_array.npy")
	for surf_num in range(len(masking_array)):
		if sopts.assigntfcesettings:
			pointer = int(sopts.assigntfcesettings[surf_num] * 2)
			tfce_settings.append(([sopts.tfce[pointer],sopts.tfce[pointer+1]]))

	if opts.outputstats:

		# load npy objects for building the tmi file
		maskname = np.load("tmi_temp/maskname.npy")
		affine_array = np.load("tmi_temp/affine_array.npy")
		vertex_array = np.load("tmi_temp/vertex_array.npy")
		face_array = np.load("tmi_temp/face_array.npy")
		surfname = np.load("tmi_temp/surfname.npy")

		for surf_num in range(len(masking_array)):
			print("Calculating stats for:\t %s" % maskname[surf_num])
			adjacency = np.load("tmi_temp/%d_adjacency_temp.npy" % surf_num)
			mask = np.load("tmi_temp/%d_mask_temp.npy" % surf_num)
			data = np.load("tmi_temp/%d_data_temp.npy" % surf_num)
			vdensity = np.load("tmi_temp/%d_vdensity_temp.npy" % surf_num)
			if not sopts.assigntfcesettings:
				calcTFCE = CreateAdjSet(float(sopts.tfce[0]), float(sopts.tfce[1]), adjacency)

			if sopts.input:
				for i, arg_pred in enumerate(sopts.input):
					if i == 0:
						pred_x = np.genfromtxt(arg_pred, delimiter=',')
					else:
						pred_x = np.column_stack([pred_x, np.genfromtxt(arg_pred, delimiter=',')])
				if sopts.assigntfcesettings:
					calcTFCE = CreateAdjSet(float(tfce_settings[surf_num][0]), float(tfce_settings[surf_num][1]), adjacency)
				temp_tvals, temp_tfce_tvals, temp_neg_tfce_tvals = low_ram_calculate_tfce(data, mask, pred_x, calcTFCE, vdensity, set_surf_count = surf_num, randomise = False, no_intercept = True)

				if surf_num == 0:
					tvals = temp_tvals
					tfce_tvals = temp_tfce_tvals
					neg_tfce_tvals = temp_neg_tfce_tvals
					if pred_x.ndim == 1: # get number of contrasts
						num_contrasts = 1
					else:
						num_contrasts = pred_x.shape[1]
				else:
					tvals = np.concatenate((tvals, temp_tvals), 1)
					tfce_tvals = np.concatenate((tfce_tvals, temp_tfce_tvals), 1)
					neg_tfce_tvals = np.concatenate((neg_tfce_tvals, temp_neg_tfce_tvals), 1)

			if sopts.inputmediation:
				medtype = sopts.inputmediation[0]
				pred_x =  np.genfromtxt(sopts.inputmediation[1], delimiter=',')
				depend_y =  np.genfromtxt(sopts.inputmediation[2], delimiter=',')
				if sopts.assigntfcesettings:
					calcTFCE = CreateAdjSet(float(tfce_settings[surf_num][0]), float(tfce_settings[surf_num][1]), adjacency)
				temp_zvals, temp_tfce_zvals = low_ram_calculate_mediation_tfce(medtype, data, mask, pred_x, depend_y, calcTFCE, vdensity, set_surf_count = surf_num, randomise = False, no_intercept = True)

				if surf_num == 0:
					zvals = temp_zvals
					tfce_zvals = temp_tfce_zvals
					if pred_x.ndim == 1: # get number of contrasts
						num_contrasts = 1
					else:
						num_contrasts = pred_x.shape[1]
				else:
					zvals = np.concatenate((zvals, temp_zvals))
					tfce_zvals = np.concatenate((tfce_zvals, temp_tfce_zvals))

		if sopts.input:
			data = np.column_stack((tvals.T, tfce_tvals.T))
			data = np.column_stack((data, neg_tfce_tvals.T))
			contrast_names = []
			for i in range(num_contrasts):
				contrast_names.append(("tstat_con%d" % (i+1)))
			for j in range(num_contrasts):
				contrast_names.append(("tstat_tfce_con%d" % (j+1)))
			for k in range(num_contrasts):
				contrast_names.append(("negtstat_tfce_con%d" % (k+1)))
		if sopts.inputmediation:
			data = np.column_stack((zvals.T, tfce_zvals.T))
			contrast_names = []
			contrast_names.append(("SobelZ"))
			contrast_names.append(("SobelZ_tfce"))

		outname = os.path.basename(str(opts.path[0]))[7:]

		if not outname.endswith('tmi'):
			outname += '.tmi'

		write_tm_filetype("%s/%s" % (os.path.dirname(str(opts.path[0])), outname), 
			image_array = data, 
			masking_array = masking_array, 
			maskname = maskname, 
			affine_array = affine_array, 
			vertex_array = vertex_array, 
			face_array = face_array, 
			surfname = surfname, 
			checkname = True, 
			columnids = np.array(contrast_names),
			tmi_history=[])

	else:
		currentTime=int(time())
		surf_num = int(opts.surfacenumber[0])
		p_range = np.array(opts.permutationrange)
		adjacency = np.load("tmi_temp/%d_adjacency_temp.npy" % surf_num)
		mask = np.load("tmi_temp/%d_mask_temp.npy" % surf_num)
		data = np.load("tmi_temp/%d_data_temp.npy" % surf_num)
		vdensity = np.load("tmi_temp/%d_vdensity_temp.npy" % surf_num)


		if sopts.assigntfcesettings:
			calcTFCE = CreateAdjSet(float(tfce_settings[surf_num][0]), float(tfce_settings[surf_num][1]), adjacency)
		else:
			calcTFCE = CreateAdjSet(float(sopts.tfce[0]), float(sopts.tfce[1]), adjacency)
		if sopts.input:
			for i, arg_pred in enumerate(sopts.input):
				if i == 0:
					pred_x = np.genfromtxt(arg_pred, delimiter=',')
				else:
					pred_x = np.column_stack([pred_x, np.genfromtxt(arg_pred, delimiter=',')])
			for perm_number in range(p_range[0],int(p_range[1]+1)):
				low_ram_calculate_tfce(data, mask, pred_x, calcTFCE, vdensity,
					set_surf_count = surf_num,
					perm_number = perm_number,
					randomise = True,
					no_intercept = True,
					output_dir = str(opts.path[0]),
					perm_seed = int(opts.seed[0]))

		if sopts.inputmediation:
			medtype = sopts.inputmediation[0]
			pred_x =  np.genfromtxt(sopts.inputmediation[1], delimiter=',')
			depend_y =  np.genfromtxt(sopts.inputmediation[2], delimiter=',')
			for perm_number in range(p_range[0],int(p_range[1]+1)):
				low_ram_calculate_mediation_tfce(medtype, data, mask, pred_x, depend_y, calcTFCE, vdensity,
					set_surf_count = surf_num,
					perm_number = perm_number,
					randomise = True,
					no_intercept = True,
					output_dir = str(opts.path[0]),
					perm_seed = int(opts.seed[0]))
		print("Mask %d, Iteration %d -> %d took %i seconds." % (surf_num, p_range[0], p_range[1], (int(time()) - currentTime)))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
