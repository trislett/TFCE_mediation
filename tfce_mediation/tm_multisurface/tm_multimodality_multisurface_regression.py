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
from tfce_mediation.tm_func import calculate_tfce, calculate_mediation_tfce, calc_mixed_tfce, apply_mfwer, create_full_mask, merge_adjacency_array, lowest_length, create_position_array, paint_surface, strip_basename, saveauto


DESCRIPTION = "MMR: Multimodality Multisurface Regression with TFCE and *.tmi formated neuroimaging files."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):

	ap.add_argument("-i_tmi", "--tmifile",
		help="Input the *.tmi file for analysis.", 
		nargs=1,
		metavar=('*.tmi'),
		required=True)

	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument("-i", "--input", 
		nargs='+', 
		help="[Predictor(s)]", 
		metavar=('*.csv'))
	group.add_argument("-im", "--inputmediation", 
		nargs=3, 
		help="[Mediation Type {I,M,Y}] [Predictor] [Dependent]", 
		metavar=('{I,M,Y}','*.csv', '*.csv'))
	ap.add_argument("-c", "--covariates", 
		nargs=1, 
		help="[Covariate(s)]", 
		metavar=('*.csv'))
	group.add_argument("-r", "--regressors", 
		nargs=1, help="Single step regression", 
		metavar=('*.csv'))
	group.add_argument("-mfwe","--multisurfacefwecorrection", 
		help="Input the stats tmi using -i_tmi *.tmi. The corrected files will be appended to the stats tmi. Note, the intercepts will be ignored.",
		action='store_true')
	group.add_argument("-medfwe","--mediationmfwe",
		nargs=1,
		choices = ['I', 'M', 'Y'],
		help="Input the stats tmi using -i_tmi *.tmi. Select the mediation type {I,M,Y}. The corrected files will be appended to the stats tmi.")

	ap.add_argument("-p", "--randomise", 
		help="Specify the range of permutations. e.g, -p 1 200", 
		nargs=2,
		type=int,
		metavar=['INT'])
	ap.add_argument("-i_name", "--analysisname",
		help="Input the *.tmi file for analysis.", 
		nargs=1)
	ap.add_argument("--tfce", 
		help="TFCE settings. H (i.e., height raised to power H), E (i.e., extent raised to power E). Default: %(default)s). H=2, E=2/3. Multiple sets of H and E values can be entered with using the -st option.", 
		nargs='+', 
		default=[2.0,0.67],
		type=float,
		metavar=('H', 'E'))
	ap.add_argument("-sa", "--setadjacencyobjs",
		help="Specify the adjaceny object to use for each mask. The number of inputs must match the number of masks in the tmi file. Note, the objects start at zero. e.g., -sa 0 1 0 1",
		nargs='+',
		type=int,
		metavar=('INT'))
	ap.add_argument("-st", "--assigntfcesettings",
		help="Specify the tfce H and E settings for each mask. -st is useful for combined analysis do voxel and vertex data. More than one set of values must inputted with --tfce. The number of inputs must match the number of masks in the tmi file. The input corresponds to each pair of --tfce setting starting at zero. e.g., -st 0 0 0 0 1 1",
		nargs='+',
		type=int,
		metavar=('INT'))
	ap.add_argument("--noweight", 
		help="Do not weight each vertex for density of vertices within the specified geodesic distance (not recommended).", 
		action="store_true")
	ap.add_argument("--subset",
		help="Analyze a subset of subjects based on a single column text file. Subset will be performed based on whether each input is finite (keep) or text (remove).", 
		nargs=1)
	ap.add_argument("--outtype", 
		help="Specify the output file type", 
		nargs='+', 
		default=['tmi'], 
		choices=('tmi', 'mgh', 'nii.gz', 'auto'))
	ap.add_argument("-cs","--concatestats", 
		help="Concantenate FWE corrected p statistic images to the stats file. Must be used with -mfwe option", 
		action="store_true")
	ap.add_argument("-l","--neglog", 
		help="Output negative log(10) pFWE corrected images (useful for visualizing effect sizes).", 
		action="store_true")
	ap.add_argument("-op", "--outputply",
		help="Projects pFWE corrected for negative and positive TFCE transformed t-statistics onto a ply mesh for visualization of results using a 3D viewer. Must be used with -mfwe option. The sigificance threshold (low and high), and either: red-yellow (r_y), blue-lightblue (b_lb) or any matplotlib colorschemes (https://matplotlib.org/examples/color/colormaps_reference.html). Note, thresholds must be postive. e.g., -op 0.95 1 r_y b_lb", 
		nargs=4, 
		metavar=('float','float', 'colormap', 'colormap'))

#	ap.add_argument("--plysmoothing",
#		help = "Apply Laplician or Taubin smoothing before visualization. Input the number of iterations (e.g., -ss 5).", 
#		nargs = 1,
#		type = int,
#		metavar = 'int')
#	ap.add_argument("--smoothingtype",
#		help = "Set type of surface smoothing to use (choices are: %(choices)s). The default is laplacian. The Taubin (aka low-pass) filter smooths curves/surfaces without the shrinkage of the laplacian filter.", 
#		nargs = 1,
#		choices = ['laplacian','taubin'],
#		default = ['laplacian'],
#		metavar = 'str')

	correctionoptions = ap.add_mutually_exclusive_group(required=False)
	correctionoptions.add_argument("-ss","--setsurface", 
		help="Must be used with -mfwe option. Input the set of surfaces to create pFWE corrected images using a range. Family-wise error rate correction will only applied to the specified surfaces. e.g., -ss 0 1 5 6", 
		nargs='+',
		type=int,
		metavar=('INT'))
	correctionoptions.add_argument("-ssr","--setsurfacerange", 
		help="Must be used with -mfwe option. Input a range to set the surfaces to create pFWE corrected images using a range. Family-wise error rate correction will only applied to the specified surfaces. e.g., -ssr 0 3", 
		nargs=2,
		type=int,
		metavar=('INT'))

	return ap

def run(opts):
	currentTime=int(time())
	if opts.multisurfacefwecorrection:
		#############################
		###### FWER CORRECTION ######
		#############################
		_, image_array, masking_array, maskname, affine_array, vertex_array, face_array, surfname, adjacency_array, tmi_history, columnids = read_tm_filetype('%s' % opts.tmifile[0], verbose=False)

		# check file dimensions
		if not image_array[0].shape[1] % 3 == 0:
			print('Print file format is not understood. Please make sure %s is statistics file.' % opts.tmifile[0])
			quit()
		else:
			num_contrasts = int(image_array[0].shape[1] / 3)

		# get surface coordinates in data array
		position_array = create_position_array(masking_array)

		if num_contrasts == 1:
			# get lists for positive and negative contrasts
			pos_range = [1]
			neg_range = [2]
		else:
			# get lists for positive and negative contrasts
			pos_range = list(range(num_contrasts, num_contrasts+num_contrasts))
			neg_range = list(range(num_contrasts*2, num_contrasts*2+num_contrasts))

		# check that randomisation has been run
		if not os.path.exists("%s/output_%s/perm_maxTFCE_surf0_tcon1.csv" % (os.getcwd(),opts.tmifile[0])): # make this safer
			print('Permutation folder not found. Please run --randomise first.')
			quit()

		#check permutation file lengths
		num_surf = len(masking_array)
		surface_range = list(range(num_surf))
		num_perm = lowest_length(num_contrasts, surface_range, opts.tmifile[0])

		if opts.setsurfacerange:
			surface_range = list(range(opts.setsurfacerange[0], opts.setsurfacerange[1]+1))
		elif opts.setsurface:
			surface_range = opts.setsurface
		if np.array(surface_range).max() > len(masking_array):
			print("Error: range does note fit the surfaces contained in the tmi file. %s contains the following surfaces" % opts.tmifile[0])
			for i in range(len(surfname)):
				print(("Surface %d : %s, %s" % (i,surfname[i], maskname[i])))
			quit()
		print("Reading %d contrast(s) from %d of %d surface(s)" % ((num_contrasts),len(surface_range), num_surf))
		print("Reading %s permutations with an accuracy of p=0.05+/-%.4f" % (num_perm,(2*(np.sqrt(0.05*0.95/num_perm)))))

		# calculate the P(FWER) images from all surfaces
		positive_data, negative_data = apply_mfwer(image_array, num_contrasts, surface_range, num_perm, num_surf, opts.tmifile[0], position_array, pos_range, neg_range, weight='logmasksize')

		# write out files
		if opts.concatestats:
			write_tm_filetype(opts.tmifile[0],
				image_array = positive_data,
				masking_array = masking_array,
				maskname = maskname,
				affine_array = affine_array,
				vertex_array = vertex_array,
				face_array = face_array,
				surfname = surfname,
				adjacency_array = adjacency_array,
				checkname = False,
				tmi_history = tmi_history)
			_, image_array, masking_array, maskname, affine_array, vertex_array, face_array, surfname, adjacency_array, tmi_history, columnids = read_tm_filetype(opts.tmifile[0], verbose=False)
			write_tm_filetype(opts.tmifile[0],
				image_array = np.column_stack((image_array[0],negative_data)),
				masking_array = masking_array,
				maskname = maskname,
				affine_array = affine_array,
				vertex_array = vertex_array,
				face_array = face_array,
				surfname = surfname,
				adjacency_array = adjacency_array,
				checkname = False,
				tmi_history = tmi_history)
		else:
			for i in range(len(opts.outtype)):
				if opts.outtype[i] == 'tmi':
					contrast_names = []

					for j in range(num_contrasts):
						contrast_names.append(("tstat_pFWER_con%d" % (j+1)))
					for k in range(num_contrasts):
						contrast_names.append(("negtstat_pFWER_con%d" % (k+1)))

					outdata = np.column_stack((positive_data,negative_data))

					if opts.neglog:
						for j in range(num_contrasts):
							contrast_names.append(("tstat_negLog_pFWER_con%d" % (j+1)))
						for k in range(num_contrasts):
							contrast_names.append(("negtstat_negLog_pFWER_con%d" % (k+1)))

					outdata = np.column_stack((outdata,-np.log10(1-positive_data)))
					outdata = np.column_stack((outdata,-np.log10(1-negative_data)))

					write_tm_filetype("pFWER_%s" % opts.tmifile[0],
						image_array = outdata,
						masking_array = masking_array,
						maskname = maskname,
						affine_array = affine_array,
						vertex_array = vertex_array,
						face_array = face_array,
						surfname = surfname,
						checkname = False,
						columnids = np.array(contrast_names),
						tmi_history = tmi_history)

				else:
					if opts.outtype[i] == 'mgh':
						savefunc = savemgh_v2
					if opts.outtype[i] == 'nii.gz':
						savefunc = savenifti_v2
					if opts.outtype[i] == 'auto':
						savefunc = saveauto
					for surf_count in surface_range:
						start = position_array[surf_count]
						end = position_array[surf_count+1]
						basename = strip_basename(maskname[surf_count])
						if not os.path.exists("output_stats"):
							os.mkdir("output_stats")
						out_image = positive_data[start:end]
						temp_image = negative_data[start:end]
						for contrast in range(num_contrasts):
							out_image[temp_image[:, contrast] != 0,contrast] = temp_image[temp_image[:, contrast] != 0,contrast] * -1
						if affine_array == []:
							savefunc(out_image,
								masking_array[surf_count],
								"output_stats/%d_%s_pFWER" % (surf_count, 
								basename))
						else:
							savefunc(out_image,masking_array[surf_count],
								"output_stats/%d_%s_pFWER" % (surf_count, basename),
								affine_array[surf_count])
						if opts.neglog:
							out_image = -np.log10(1 - positive_data[start:end,contrast])
							temp_image = np.log10(1 - negative_data[start:end,contrast])
							for contrast in range(num_contrasts):
								out_image[temp_image[:, contrast] != 0,contrast] = temp_image[temp_image[:, contrast] != 0,contrast]
							if affine_array == []:
								savefunc(out_image,
									masking_array[surf_count],
									"output_stats/%d_%s_negLog_pFWER" % (surf_count, basename))
							else:
								savefunc(out_image,
									masking_array[surf_count],
									"output_stats/%d_%s_negLog_pFWER" % (surf_count, basename),
									affine_array[surf_count])
		if opts.outputply:
			colorbar = True
			if not os.path.exists("output_ply"):
				os.mkdir("output_ply")
			for contrast in range(num_contrasts):
				for surf_count in surface_range:
					start = position_array[surf_count]
					end = position_array[surf_count+1]
					basename = strip_basename(maskname[surf_count])
					if masking_array[surf_count].shape[2] > 1:
						img_data = np.zeros((masking_array[surf_count].shape))
						combined_data = positive_data[start:end,contrast]
						combined_data[combined_data<=0] = negative_data[start:end,contrast][combined_data<=0] * -1
						combined_data[np.abs(combined_data)<float(opts.outputply[0])] = 0
						img_data[masking_array[surf_count]] = combined_data
						v, f, values = convert_voxel(img_data, affine = affine_array[surf_count], absthreshold = float(opts.outputply[0]))
						if not v == []:
							out_color_array = paint_surface(opts.outputply[0],
								opts.outputply[1],
								opts.outputply[2],
								values,
								save_colorbar=colorbar)
							negvalues = values * -1
							index = negvalues > float(opts.outputply[0])
							out_color_array2 = paint_surface(opts.outputply[0],
								opts.outputply[1],
								opts.outputply[3],
								negvalues,
								save_colorbar=colorbar)
							out_color_array[index,:] = out_color_array2[index,:]
							save_ply(v,f, "output_ply/%d_%s_pFWE_tcon%d.ply" % (surf_count, basename, contrast+1), out_color_array)
							colorbar = False
						else:
							print("No output for %d %s T-contrast %d" % (surf_count, basename, contrast+1))
					else:
						img_data = np.zeros((masking_array[surf_count].shape[0]))
						img_data[masking_array[surf_count][:,0,0]==True] = positive_data[start:end,contrast]
						out_color_array = paint_surface(opts.outputply[0],
							opts.outputply[1],
							opts.outputply[2],
							img_data,
							save_colorbar=colorbar)
						img_data[masking_array[surf_count][:,0,0]==True] = negative_data[start:end,contrast]
						index = img_data > float(opts.outputply[0])
						out_color_array2 = paint_surface(opts.outputply[0], 
							opts.outputply[1],
							opts.outputply[3],
							img_data,
							save_colorbar=colorbar)
						out_color_array[index,:] = out_color_array2[index,:]
						save_ply(vertex_array[surf_count],
							face_array[surf_count],
							"output_ply/%d_%s_pFWE_tcon%d.ply" % (surf_count, basename, contrast+1),
							out_color_array)
						colorbar = False

	elif opts.mediationmfwe: # temporary solution -> maybe a general function instead of bulky code

		_, image_array, masking_array, maskname, affine_array, vertex_array, face_array, surfname, adjacency_array, tmi_history, columnids = read_tm_filetype('%s' % opts.tmifile[0], verbose=False)

		# check file dimensions
		if not image_array[0].shape[1] % 2 == 0:
			print('Print file format is not understood. Please make sure %s is statistics file.' % opts.tmifile[0])
			quit()

		# get surface coordinates in data array
		position_array = create_position_array(masking_array)


		# check that randomisation has been run
		if not os.path.exists("%s/output_%s/perm_maxTFCE_surf0_%s_zstat.csv" % (os.getcwd(),opts.tmifile[0], opts.mediationmfwe[0])): # make this safer
			print('Permutation folder not found. Please run --randomise first.')
			quit()


		#check permutation file lengths
		num_surf = len(masking_array)
		surface_range = list(range(num_surf))
		num_perm = lowest_length(1, surface_range, opts.tmifile[0], medtype = opts.mediationmfwe[0])

		if opts.setsurfacerange:
			surface_range = list(range(opts.setsurfacerange[0], opts.setsurfacerange[1]+1))
		elif opts.setsurface:
			surface_range = opts.setsurface
		if np.array(surface_range).max() > len(masking_array):
			print("Error: range does note fit the surfaces contained in the tmi file. %s contains the following surfaces" % opts.tmifile[0])
			for i in range(len(surfname)):
				print(("Surface %d : %s, %s" % (i,surfname[i], maskname[i])))
			quit()
		print("Reading %d contrast(s) from %d of %d surface(s)" % (1,len(surface_range), num_surf))
		print("Reading %s permutations with an accuracy of p=0.05+/-%.4f" % (num_perm,(2*(np.sqrt(0.05*0.95/num_perm)))))

		# calculate the P(FWER) images from all surfaces
		positive_data = apply_mfwer(image_array, 1, surface_range, num_perm, num_surf, opts.tmifile[0], position_array, [1], weight='logmasksize', mediation = True, medtype = opts.mediationmfwe[0])
		if opts.outtype[0] == 'tmi':
			contrast_names = []

			contrast_names.append(("zstat_pFWER"))
			outdata = positive_data

			if opts.neglog:
				contrast_names.append(("zstat_negLog_pFWER"))
				outdata = np.column_stack((outdata,-np.log10(1-positive_data)))


			write_tm_filetype("pFWER_%s_%s" % (opts.mediationmfwe[0], opts.tmifile[0]),
				image_array = outdata,
				masking_array = masking_array,
				maskname = maskname,
				affine_array = affine_array,
				vertex_array = vertex_array,
				face_array = face_array,
				surfname = surfname,
				checkname = False,
				columnids = np.array(contrast_names),
				tmi_history = tmi_history)


	else:
		##################################
		###### STATISTICAL ANALYSIS ######
		##################################
		# read tmi file
		if opts.randomise:
			_, image_array, masking_array, _, _, _, _, _, adjacency_array, _, _  = read_tm_filetype(opts.tmifile[0])
			_ = None
		else: 
			element, image_array, masking_array, maskname, affine_array, vertex_array, face_array, surfname, adjacency_array, tmi_history, _  = read_tm_filetype(opts.tmifile[0])
		# get surface coordinates in data array
		position_array = create_position_array(masking_array)

		if opts.setadjacencyobjs:
			if len(opts.setadjacencyobjs) == len(masking_array):
				adjacent_range = np.array(opts.setadjacencyobjs, dtype = np.int)
			else:
				print("Error: # of masking arrays (%d) must and list of matching adjacency (%d) must be equal." % (len(masking_array), len(opts.setadjacencyobjs)))
				quit()
		else: 
			adjacent_range = list(range(len(adjacency_array)))
		calcTFCE = []
		if opts.assigntfcesettings:
			if not len(opts.assigntfcesettings) == len(masking_array):
				print("Error: # of masking arrays (%d) must and list of matching tfce setting (%d) must be equal." % (len(masking_array), len(opts.assigntfcesettings)))
				quit()
			if not len(opts.tfce) % 2 == 0:
				print("Error. The must be an even number of input for --tfce")
				quit()
			tfce_settings_mask = []
			for i in np.unique(opts.assigntfcesettings):
				tfce_settings_mask.append((np.array(opts.assigntfcesettings) == int(i)))
				pointer = int(i*2)
				adjacency = merge_adjacency_array(np.array(adjacent_range)[tfce_settings_mask[int(i)]], np.array(adjacency_array)[tfce_settings_mask[int(i)]])
				calcTFCE.append((CreateAdjSet(float(opts.tfce[pointer]), float(opts.tfce[pointer+1]), adjacency)))
				del adjacency
		else:
			adjacency = merge_adjacency_array(adjacent_range, adjacency_array)
			calcTFCE.append((CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjacency)))

		# make mega mask
		fullmask = create_full_mask(masking_array)

		if not opts.noweight:
			# correction for vertex density
			vdensity = []
			#np.ones_like(masking_array)
			for i in range(len(masking_array)):
				temp_vdensity = np.zeros((adjacency_array[adjacent_range[i]].shape[0]))
				for j in range(adjacency_array[adjacent_range[i]].shape[0]):
					temp_vdensity[j] = len(adjacency_array[adjacent_range[i]][j])
				if masking_array[i].shape[2] == 1:
					temp_vdensity = temp_vdensity[masking_array[i][:,0,0]==True]
				vdensity = np.hstack((vdensity, np.array((1 - (temp_vdensity/temp_vdensity.max())+(temp_vdensity.mean()/temp_vdensity.max())), dtype=np.float32)))
			del temp_vdensity
		else:
			vdensity = 1

		#load regressors
		if opts.input:
			for i, arg_pred in enumerate(opts.input):
				if i == 0:
					pred_x = np.genfromtxt(arg_pred, delimiter=',')
				else:
					pred_x = np.column_stack([pred_x, np.genfromtxt(arg_pred, delimiter=',')])

			if opts.covariates:
				covars = np.genfromtxt(opts.covariates[0], delimiter=',')
				x_covars = np.column_stack([np.ones(len(covars)),covars])
			if opts.subset:
				masking_variable = np.isfinite(np.genfromtxt(str(opts.subset[0]), delimiter=','))
				if opts.covariates:
					merge_y = resid_covars(x_covars,image_array[0][:,masking_variable])
				else:
					merge_y = image_array[0][:,masking_variable].T 
					print("Check dimensions") # CHECK
					print(merge_y.shape)
			else:
				if opts.covariates:
					merge_y = resid_covars(x_covars,image_array[0])
				else:
					merge_y = image_array[0].T
		if opts.inputmediation:
			medtype = opts.inputmediation[0]
			pred_x =  np.genfromtxt(opts.inputmediation[1], delimiter=',')
			depend_y =  np.genfromtxt(opts.inputmediation[2], delimiter=',')
			if opts.covariates:
				covars = np.genfromtxt(opts.covariates[0], delimiter=',')
				x_covars = np.column_stack([np.ones(len(covars)), covars])
				merge_y = resid_covars(x_covars, image_array[0])
			else:
				merge_y = image_array[0].T

		if opts.regressors:
			arg_predictor = opts.regressors[0]
			pred_x = np.genfromtxt(arg_predictor, delimiter=',')

			if opts.subset:
				masking_variable = np.isfinite(np.genfromtxt(str(opts.subset[0]), delimiter=','))
				merge_y=image_array[0][:,masking_variable].T
			else:
				merge_y=image_array[0].T

		# cleanup 
		image_array = None
		adjacency_array = None
		adjacency = None

		if opts.analysisname:
			outname = opts.analysisname[0]
		else:
			outname = opts.tmifile[0][:-4]

		# make output folder
		if not os.path.exists("output_%s" % (outname)):
			os.mkdir("output_%s" % (outname))
		os.chdir("output_%s" % (outname))
		if opts.randomise:
			randTime=int(time())
			mapped_y = merge_y.astype(np.float32, order = "C") # removed memory mapping
			merge_y = None
			if not outname.endswith('tmi'):
				outname += '.tmi'

			if opts.inputmediation:
				outname = 'med_stats_' + outname
			else:
				outname = 'stats_' + outname

			if not os.path.exists("output_%s" % (outname)):
				os.mkdir("output_%s" % (outname))
			os.chdir("output_%s" % (outname))
			for i in range(opts.randomise[0],(opts.randomise[1]+1)):
				if opts.assigntfcesettings:
					calc_mixed_tfce(opts.assigntfcesettings, 
						mapped_y,
						masking_array,
						position_array,
						vdensity,
						pred_x,
						calcTFCE,
						perm_number=i,
						randomise = True)
				elif opts.inputmediation:
					calculate_mediation_tfce(medtype,
						mapped_y,
						masking_array,
						pred_x,
						depend_y,
						calcTFCE[0],
						vdensity,
						position_array,
						fullmask,
						perm_number = i,
						randomise = True)
				else:
					calculate_tfce(mapped_y, 
						masking_array,
						pred_x,
						calcTFCE[0],
						vdensity,
						position_array,
						fullmask,
						perm_number=i,
						randomise = True)
			print(("Total time took %.1f seconds" % (time() - currentTime)))
			print(("Randomization took %.1f seconds" % (time() - randTime)))
		else:
			# Run TFCE
			if opts.assigntfcesettings:
				tvals, tfce_tvals, neg_tfce_tvals = calc_mixed_tfce(opts.assigntfcesettings,
					merge_y,
					masking_array,
					position_array,
					vdensity,
					pred_x,
					calcTFCE)
			elif opts.inputmediation:
				SobelZ, tfce_SobelZ = calculate_mediation_tfce(medtype,
					merge_y,
					masking_array,
					pred_x,
					depend_y,
					calcTFCE[0],
					vdensity,
					position_array,
					fullmask)
			else:
				tvals, tfce_tvals, neg_tfce_tvals = calculate_tfce(merge_y,
					masking_array,
					pred_x, calcTFCE[0],
					vdensity,
					position_array,
					fullmask)
			if opts.outtype[0] == 'tmi':
				if not outname.endswith('tmi'):
					outname += '.tmi'
				if opts.inputmediation:
					outname = 'med_stats_' + outname
				else:
					outname = 'stats_' + outname

				if opts.inputmediation:
					contrast_names = []
					contrast_names.append(("SobelZ"))
					contrast_names.append(("SobelZ_tfce"))
					outdata = np.column_stack((SobelZ.T, tfce_SobelZ.T))
				else:
					if tvals.ndim == 1:
						num_contrasts = 1
					else:
						num_contrasts = tvals.shape[0]

					contrast_names = []
					for i in range(num_contrasts):
						contrast_names.append(("tstat_con%d" % (i+1)))
					for j in range(num_contrasts):
						contrast_names.append(("tstat_tfce_con%d" % (j+1)))
					for k in range(num_contrasts):
						contrast_names.append(("negtstat_tfce_con%d" % (k+1)))
					outdata = np.column_stack((tvals.T, tfce_tvals.T))
					outdata = np.column_stack((outdata, neg_tfce_tvals.T))

				# write tstat
				write_tm_filetype(outname, 
					image_array = outdata, 
					masking_array = masking_array, 
					maskname = maskname, 
					affine_array = affine_array, 
					vertex_array = vertex_array, 
					face_array = face_array, 
					surfname = surfname, 
					checkname = False, 
					columnids = np.array(contrast_names),
					tmi_history=[])

			else:
				print("not implemented yet")

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
