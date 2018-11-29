#!/usr/bin/env python

#    GLM models including within-subject effect and TFCE
#    Copyright (C) 2018  Tristram Lett

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
import nibabel as nib
import argparse as ap
import pandas as pd

from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.pyfunc import write_vertStat_img, write_voxelStat_img, create_adjac_vertex, create_adjac_voxel, reg_rm_ancova_one_bs_factor, reg_rm_ancova_two_bs_factor, glm_typeI, calc_indirect, dummy_code, column_product, stack_ones, import_voxel_neuroimage

def check_columns(pdData, datatype, folders = None, surface = None, FWHM = None, filelists = None, voximgs = None, tmi = None, tempdir = None):
	for counter, roi in enumerate(pdData.columns):
		if counter == 0:
			num_subjects = len(pdData[roi])
		a = np.unique(pdData[roi])
		num_missing = np.sum(pdData[roi].isnull()*1)
		if len(a) > 10:
			astr = '[n>10]'
		else:
			astr = ','.join(a.astype(np.str))
			astr = '['+astr+']'
		if num_missing == 0:
			print("[%d] : %s\t%s" % (counter, roi, astr))
		else:
			print("[%d] : %s\t%s\t\tCONTAINS %d MISSING VARIABLES!" % (counter, roi, astr, num_missing))

	print("\nChecking image lengths [# subjects = %d]" % num_subjects)
	if datatype == 'surface':
		for sf in folders:
			temp_img = ("%s/lh.all.%s.%s.mgh" % (sf,surface,FWHM))
			if not os.path.isfile(temp_img):
				print ("Error: %s not found." % temp_img)
			else:
				temp_num_img = nib.freesurfer.mghformat.load(temp_img).shape[-1]
				if temp_num_img == num_subjects:
					print("%s ...OK" % temp_img)
				else:
					print("Error: Length of %s [%d] does not match number of subjects[%d]" % (temp_img, temp_num_img, num_subjects))
				temp_img = ("%s/rh.all.%s.%s.mgh" % (sf,surface,FWHM))
				temp_num_img = nib.freesurfer.mghformat.load(temp_img).shape[-1]
				if temp_num_img == num_subjects:
					print("%s ...OK" % temp_img)
				else:
					print("Error: Length of %s [%d] does not match number of subjects[%d]" % (temp_img, temp_num_img, num_subjects))
	if datatype == 'volumetric':
		for temp_img in voximgs:
			if not os.path.isfile(temp_img):
				print ("Error: %s not found." % temp_img)
			elif len(nib.load(temp_img).shape) != 4:
				print ("Error: %s is not a 4D image." % temp_img)
			else:
				temp_num_img = nib.load(temp_img).shape[-1]
				if temp_num_img == num_subjects:
					print("%s ...OK" % temp_img)
				else:
					print("Error: Length of %s [%d] does not match number of subjects[%d]" % (temp_img, temp_num_img, num_subjects))
				if temp_num_img == num_subjects:
					print("%s ...OK" % temp_img)
				else:
					print("Error: Length of %s [%d] does not match number of subjects[%d]" % (temp_img, temp_num_img, num_subjects))
	if datatype == 'filelist':
		for filelist in filelists:
			if not os.path.isfile(filelist):
				print ("Error: %s not found." % filelist)
			loc_arr = np.genfromtxt(filelist, dtype=str)
			if len(loc_arr) == num_subjects:
				print("%s ...EXISTS" % filelist)
			num_missing = 0
			for img in loc_arr:
				if not os.path.isfile(filelist):
					num_missing +=1
			if num_missing == 0:
				print("%s ...FOUND ALL IMAGES" % filelist)
			else:
				print ("Error: %d image not found in %s." % (num_missing,filelist))
	if datatype == 'tmi':
		print("TMI not supported yet.")
	if datatype == 'tmp_folder':
		data = np.load("%s/nonzero.npy" % tempdir)
		temp_num_img = data.shape[1]
		if temp_num_img == num_subjects:
			print("%s/nonzero.npy ...OK" % tempdir)
		else:
			print data.shape
			print("Error: Length of nonzero.npy [%d] does not match number of subjects[%d]" % (temp_num_img, num_subjects))


def load_vars(pdCSV, variables, exog = [], names = []):
	if len(variables) % 2 == 1:
		print("Error: each input must be followed by data type. e.g., -glm age c sex d site d (d = discrete, c = continous)")
	num_exog = len(variables) / 2
	for i in range(num_exog):
		j = i * 2 
		k = j + 1
		if variables[k] == 'c':
			print("Coding %s as continous variable" % variables[j])
			temp = dummy_code(np.array(pdCSV[variables[j]]), iscontinous = True)
			temp = temp[:,np.newaxis]
			exog.append(temp)
		elif variables[k] == 'd':
			print("Coding %s as discrete variable" % variables[j])
			temp = dummy_code(np.array(pdCSV[variables[j]]), iscontinous = False)
			if temp.ndim == 1:
				temp = temp[:,np.newaxis]
			exog.append(temp)
		else:
			print("Error: variable type is not understood")
		names.append(variables[j])
	return (exog, names)

def load_interactions(intvariables, varnames = [], exog = [], covarnames = [], covars = []):
	for int_terms in intvariables:
		interaction_vars = int_terms.split("*")
		int_name = ".X.".join(interaction_vars)
		if interaction_vars[0] in varnames:
			for i, scale_var in enumerate(interaction_vars):
				if i == 0:
					int_temp = exog[varnames.index(interaction_vars[i])]
				else:
					int_temp = column_product(int_temp, exog[varnames.index(interaction_vars[i])])
			exog.append(int_temp)
			varnames.append(int_name)
		elif interaction_vars[0] in covarnames:
			for i, scale_var in enumerate(interaction_vars):
				if i == 0:
					int_temp = covars[covarnames.index(interaction_vars[i])]
				else:
					int_temp = column_product(int_temp, covars[covarnames.index(interaction_vars[i])])
			covars.append(int_temp)
			covarnames.append(int_name)
		else:
			print("Error: interaction variables must be contained in -glm or -c")
	return (varnames, exog, covarnames, covars)

# statistical models (glm, mediation, rmancova_one, rmancova_two)
def save_temporary_files(statmodel, modality_type = None, **kwargs):
	if statmodel == "glm":
		tempdir = "tmtemp_GLM"
	if statmodel == "mediation":
		tempdir = "tmtemp_mediation"
	if statmodel == "rmancova_one":
		tempdir = "tmtemp_rmANCOVA1BS"
	if statmodel == "rmancova_two":
		tempdir = "tmtemp_rmANCOVA2BS"
	if modality_type is not None:
		tempdir += "_%s" % modality_type
	#save variables
	if not os.path.exists(tempdir):
		os.mkdir(tempdir)
	for save_item in kwargs:
		np.save("%s/%s" % (tempdir, save_item),kwargs[save_item])

DESCRIPTION = "Vertex-wise GLMs include within-subject interactions with TFCE."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):

	ap.add_argument("-i", "--inputcsv", 
		nargs=1, 
		help="Input folder containing each surface interval.", 
		metavar=('*.csv'),
		required=True)
	ap.add_argument("-sc", "--subjectidcolumns",
		help="Select the subject id column for merging *.csv files. Default: %(default)s)",
		nargs=1,
		default=['SubjID'])
	ap.add_argument("-on", "--outputcolumnnames", 
		help="Outputs the input CSV column names, and check the column length and number of images.", 
		action='store_true')

	imputtypes = ap.add_mutually_exclusive_group(required=True)
	imputtypes.add_argument("-si", "--surfaceinputfolder", 
		nargs='+', 
		help="Input folder containing each surface interval. -si {folder1} .. {foldern}", 
		metavar=('PATH/TO/DIR'))
	imputtypes.add_argument("-vi", "--volumetricinputs", 
		nargs='+', 
		help="4D input files for each timepoint (nifti, mgh, and minc are supported). -vi {image0} .. {imageN}", 
		metavar=('PATH/TO/IMAGE_FILE'))
	imputtypes.add_argument("-vil", "--volumetricinputlist",
		nargs='+', 
		help="Input text file(s) contains a list of voxelimages (nifti, mgh, and minc are supported). Each input is consider to be a new interval. -vif {images0.txt} ... {imagesN.txt}", 
		metavar=('PATH/TO/DIR'))
	imputtypes.add_argument("-ti", "--tmiinputs", 
		nargs='+', 
		help="Input folder containing each surface interval. -si {folder1} .. {foldern}", 
		metavar=('PATH/TO/data.tmi'))
	imputtypes.add_argument("-tmp", "--usetemporaryfolder", 
		nargs=1, 
		help="Import data from an existing temporary folder. This skips the need to re-import the data files while performing other statistical analyses.", 
		metavar=('PATH/TO/tmtemp_{*}'))

	ap.add_argument("-s", "--surface",  
		nargs=1, 
		metavar=('{surface}'), 
		required=False)

	ap.add_argument("-f", "--fwhm", 
		help="Specific all surface file with different smoothing. Default is 03B (recommended)" , 
		nargs=1, 
		default=['03B'], 
		metavar=('??B'))

	stat = ap.add_mutually_exclusive_group(required=False)
	stat.add_argument("-glm","--generalizedlinearmodel",
		nargs='+',
		help="Generalized linear model that uses type I sum of squares. Each exogenous variable must be specified as either discrete (d) or continous (c). Output metrics can be specified by -gs as either F-statics, T-statistics, or all. Interactions can be specified using -ei. Only one interval is supported. e.g., -glm sex d age c genotype d. [-glm {exogenous_1} {type_1} ... {exogenous_k} {type_k}]",
		metavar=('exog1', '{d|c}'))
	stat.add_argument("-ofa","--onebetweenssubjectfactor",
		nargs=2,
		help="ANCOVA with one between-subject (fixed) factor and neuroimage as the within-subject (random) factor (mixed-effect model). AKA, the repeated-measure ANCOVA. Covariates may also be included. Interactions will be coded automatically (Factor1*Time). ANOVA uses type I sum of squares (order matters). The between-subject factor must be specified as either discrete (d) or continous (c). Additional interactions among the covariates can be specified using -ei. e.g., -ofa genotype d. [-ofa {factor} {type}]",
		metavar=('exog1', '{d|c}'))
	stat.add_argument("-tfa","--twobetweenssubjectfactor",
		nargs=4,
		help="ANCOVA with two between-subject (fixed) factors and neuroimage as the within-subject (random) factor (mixed-effect model). AKA, the repeated-measure ANCOVA. Covariates may also be included. Interactions will be coded automatically (F1*F2, F1*Time,F2xTime, F1*F2*Time). ANOVA uses type I sum of squares (order matters). The between-subject factor must be specified as either discrete (d) or continous (c). Additional interactions among the covariates can be specified using -ei. e.g., -tfa sex d genotype d. [-tfa {factor1} {type1} {factor2} {type2}]",
		metavar=('exog1', '{d|c}', 'exog2', '{d|c}'))
	stat.add_argument("-med","--mediation",
		nargs=5,
		help="Mediation. Only one interval is currently supported.",
		metavar=('{I|M|Y}', 'left_var', '{d|c}', 'right_var', '{d|c}'))

	ap.add_argument("-c", "--covariates",
		help="Covariates of no interest.",
		nargs='+',
		metavar=('exogn', '{d|c}'),
		required=False)
	ap.add_argument("-ei", "--exogenousvariableinteraction",
		help="Specify interactions. The variables must be exognenous variables in either -c or -glm (e.g.-ei site*scanner sex*age age*age age*age*age) [-ei {exogi*exogj}...]",
		nargs='+',
		metavar=('exogn'),
		required=False)
	ap.add_argument("-gs","--glmoutputstatistic",
		default = ['f'],
		choices = ['f','t', 'all'])

	mask = ap.add_mutually_exclusive_group(required=False)
	mask.add_argument("-m","--binarymask", 
		help="Load binary mask surface for lh and rh, respectively.", 
		nargs='+', 
		metavar=('{nii|nii.gz|mnc|mgh|mgz}'))
	mask.add_argument("-fm","--fsmask", 
		help="Create masking array based on fsaverage label. Default is cortex", 
		const='cortex', 
		nargs='?')
	mask.add_argument("-l","--label", 
		help="Load label as masking array for lh and rh, respectively.", 
		nargs=2, 
		metavar=('lh.*.label','rh.*.label'))

	adjac = ap.add_mutually_exclusive_group(required=False)
	adjac.add_argument("-a", "--adjfiles", help="Load custom adjacency set. For surface data an adjacent set is required for each hemisphere.", 
		nargs='+', 
		metavar=('*.npy'))
	adjac.add_argument("-d", "--vertexdist", 
		help="Load supplied adjacency sets geodesic distance in mm. Default is 3 (recommended).", 
		choices = [1,2,3], 
		type=int,  
		nargs=1, 
		default=[3])
	adjac.add_argument("-t", "--vertextriangularmesh", 
		help="Create adjacency based on triangular mesh without specifying distance.",
		action='store_true')
	ap.add_argument("--vertexsrf", 
		help="Load surfaces for triangular mesh tfce. --vertextriangularmesh option must be used.",
		nargs=2,
		metavar=('lh.*.srf', 'rh.*.srf'))
	ap.add_argument("--tfce", 
		help="TFCE settings. H (i.e., height raised to power H), E (i.e., extent raised to power E). Default: %(default)s). H=2, E=2/3 is the point at which the cummulative density function is approximately Gaussian distributed.", 
		nargs=3, 
		default=[2, 0.67, 26], 
		metavar=('H', 'E', 'Adj'))
	ap.add_argument("--noweight", 
		help="Do not weight each vertex for density of vertices within the specified geodesic distance.", 
		action="store_true")
	ap.add_argument("--noreducedmodel", 
		help="Do not use a reduced model for permutation testing (i.e., the residuals after controlling for between-subject effects).",
		action="store_true")
	return ap

def run(opts):

	scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	CSV = opts.inputcsv[0]
	pdCSV = pd.read_csv(CSV, delimiter=',', index_col=None)
	optstfce = opts.tfce

	imgext = '.nii.gz' # temporary

	# output column/variable names.
	if opts.outputcolumnnames:
		if opts.surfaceinputfolder:
			check_columns(pdCSV, datatype = 'surface', folders = opts.surfaceinputfolder, surface = opts.surface[0], FWHM = opts.fwhm[0])
		if opts.volumetricinputs:
			check_columns(pdCSV, datatype = 'volumetric', voximgs = opts.volumetricinputs)
		if opts.volumetricinputlist:
			check_columns(pdCSV, datatype = 'filelist', filelists = opts.volumetricinputlist)
		if opts.tmiinputs:
			check_columns(pdCSV, datatype = 'tmi', tmi = opts.tmiinputs)
		if opts.usetemporaryfolder:
			check_columns(pdCSV, datatype = 'tmp_folder', tempdir = opts.usetemporaryfolder[0])
		quit()
	else:
		# secondary required variables
		if opts.generalizedlinearmodel:
			pass
		elif opts.onebetweenssubjectfactor:
			pass
		elif opts.twobetweenssubjectfactor:
			pass
		elif opts.mediation:
			pass
		else:
			print("ERROR: please specify statistical model {-glm | -of | -tw | -med}")
			quit()

	data = []
	if opts.surfaceinputfolder:
		if not opts.surface:
			print("ERROR: surface type must be specified. -s {surface}")
			quit()
		surface = opts.surface[0]
		FWHM = opts.fwhm[0]
		folders = opts.surfaceinputfolder

		#load surface data
		for i, sfolder in enumerate(folders):
			if i == 0:
				img_data_lh = nib.freesurfer.mghformat.load("%s/lh.all.%s.%s.mgh" % (sfolder, surface,FWHM))
				data_full_lh = img_data_lh.get_data()
				data_lh = np.squeeze(data_full_lh)
				affine_mask_lh = img_data_lh.affine
				outdata_mask_lh = np.zeros_like(data_full_lh[:,:,:,1])
				mask_lh = data_lh.mean(1)!=0
				img_data_rh = nib.freesurfer.mghformat.load("%s/rh.all.%s.%s.mgh" % (sfolder, surface,FWHM))
				data_full_rh = img_data_rh.get_data()
				data_rh = np.squeeze(data_full_rh)
				affine_mask_rh = img_data_rh.affine
				outdata_mask_rh = np.zeros_like(data_full_rh[:,:,:,1])
				mask_rh = data_rh.mean(1)!=0
				data.append(np.hstack((data_lh[mask_lh].T,data_rh[mask_rh].T)))
				num_vertex_lh = data_lh[mask_lh].shape[0]
				all_vertex = data_full_lh.shape[0]
			else:
				data_lh = np.squeeze(nib.freesurfer.mghformat.load("%s/lh.all.%s.%s.mgh" % (sfolder, surface,FWHM)).get_data())
				data_rh = np.squeeze(nib.freesurfer.mghformat.load("%s/rh.all.%s.%s.mgh" % (sfolder, surface,FWHM)).get_data())
				data.append(np.hstack((data_lh[mask_lh].T,data_rh[mask_rh].T)))
				data_lh = data_rh = []
		data = np.array(data)
		nonzero = np.empty_like(data)
		nonzero[:] = np.copy(data)

		#TFCE
		if opts.vertextriangularmesh:
			# 3 Neighbour vertex connectity
			print("Creating adjacency set")
			if opts.vertexsrf:
				v_lh, faces_lh = nib.freesurfer.read_geometry(opts.vertexsrf[0])
				v_rh, faces_rh = nib.freesurfer.read_geometry(opts.vertexsrf[1])
			else:
				v_lh, faces_lh = nib.freesurfer.read_geometry("%s/fsaverage/surf/lh.sphere" % os.environ["SUBJECTS_DIR"])
				v_rh, faces_rh = nib.freesurfer.read_geometry("%s/fsaverage/surf/rh.sphere" % os.environ["SUBJECTS_DIR"])
			adjac_lh = create_adjac_vertex(v_lh,faces_lh)
			adjac_rh = create_adjac_vertex(v_rh,faces_rh)
		elif opts.adjfiles:
			print("Loading prior adjacency set")
			arg_adjac_lh = opts.adjfiles[0]
			arg_adjac_rh = opts.adjfiles[1]
			adjac_lh = np.load(arg_adjac_lh)
			adjac_rh = np.load(arg_adjac_rh)
		elif opts.vertexdist:
			print("Loading prior adjacency set for %s mm" % opts.vertexdist[0])
			adjac_lh = np.load("%s/adjacency_sets/lh_adjacency_dist_%s.0_mm.npy" % (scriptwd,str(opts.vertexdist[0])))
			adjac_rh = np.load("%s/adjacency_sets/rh_adjacency_dist_%s.0_mm.npy" % (scriptwd,str(opts.vertexdist[0])))
		else:
			print("Error")
		if opts.noweight or opts.vertextriangularmesh:
			vdensity_lh = 1
			vdensity_rh = 1
		else:
			# correction for vertex density
			vdensity_lh = np.zeros((adjac_lh.shape[0]))
			vdensity_rh = np.zeros((adjac_rh.shape[0]))
			for i in range(adjac_lh.shape[0]):
				vdensity_lh[i] = len(adjac_lh[i])
			for j in range(adjac_rh.shape[0]): 
				vdensity_rh[j] = len(adjac_rh[j])
			vdensity_lh = np.array((1 - (vdensity_lh/vdensity_lh.max()) + (vdensity_lh.mean()/vdensity_lh.max())), dtype=np.float32)
			vdensity_rh = np.array((1 - (vdensity_rh/vdensity_rh.max()) + (vdensity_rh.mean()/vdensity_rh.max())), dtype=np.float32)
		calcTFCE_lh = CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjac_lh)
		calcTFCE_rh = CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjac_rh)

	if opts.volumetricinputs:
		images = opts.volumetricinputs
		for i, vimage in enumerate(images):
			if i == 0:
				if opts.binarymask:
					img_mask, data_mask = import_voxel_neuroimage(opts.binarymask[0])
					affine_mask = img_mask.affine
					mask_index = data_mask > 0.99
					tempdata = import_voxel_neuroimage(vimage, mask_index)
					data.append(tempdata.T)
				else:
					img, img_data = import_voxel_neuroimage(vimage)
					affine_mask = img.affine
					data_mask = np.zeros_like(img_data[:,:,:,1])
					mask_index = img_data.mean(3)!=0
					data_mask[mask_index] = 1
					tempdata = img_data[mask_index]
					data.append(tempdata.T)
			else:
				tempdata = import_voxel_neuroimage(vimage, mask_index)
				data.append(tempdata.T)
		data = np.array(data)
		print data.shape
		nonzero = np.zeros_like(data)
		nonzero[:] = np.copy(data)
		print nonzero.shape
		#TFCE
		if opts.adjfiles:
			print("Loading prior adjacency set")
			arg_adjac_lh = opts.adjfiles[0]
			adjac = np.load(arg_adjac_lh)
		else:
			adjac = create_adjac_voxel(mask_index, data_mask, len(data_mask[mask_index]), dirtype = opts.tfce[2])
		calcTFCE = CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjac) # H=2, E=2, 26 neighbour connectivity
	if opts.volumetricinputlist:
		img_lists = opts.volumetricinputlist
		for i in range(len(img_lists)):
			img_data = []
			if i == 0:
				if opts.binarymask:
					img_mask, data_mask = import_voxel_neuroimage(opts.binarymask[0])
					affine_mask = img_mask.affine
					mask_index = data_mask > 0.99
					for image_path in np.genfromtxt(img_lists[i], delimiter=',', dtype=str):
						img_data.append(import_voxel_neuroimage(image_path, mask_index))
					data.append(np.array(img_data))
					del img_data
				else:
					temp, temp_data = import_voxel_neuroimage(np.genfromtxt(img_lists[i], delimiter=',', dtype=str)[0]) # temporarly grab the first img
					print("WARNING: only the first image was used to create a mask. It is recommended to use a binary mask (-m) with -vil")
					affine_mask = temp.affine
					mask_index = temp_data > 0.99
					for image_path in np.genfromtxt(img_lists[i], delimiter=',', dtype=str):
						img_data.append(import_voxel_neuroimage(image_path, mask_index))
					img_data = np.array(img_data)
					data_mask = np.zeros_like(temp_data)
					data_mask[mask_index] = 1
					data.append(np.array(img_data))
					del temp_data
					del img_data
			else:
				for image_path in np.genfromtxt(img_lists[i], delimiter=',', dtype=str):
					img_data.append(import_voxel_neuroimage(image_path, mask_index))
				data.append(np.array(img_data))
		data = np.array(data)
		nonzero = np.empty_like(data)
		nonzero[:] = np.copy(data)
		print ("Voxel data loaded [%d intervals, %d subjects, %d voxels]" % (data.shape[0], data.shape[1], data.shape[2]))
		#TFCE
		if opts.adjfiles:
			print("Loading prior adjacency set")
			arg_adjac_lh = opts.adjfiles[0]
			adjac = np.load(arg_adjac_lh)
		else:
			adjac = create_adjac_voxel(mask_index, data_mask, data.shape[2], dirtype=opts.tfce[2])
		calcTFCE = CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjac) # H=2, E=2, 26 neighbour connectivity
	if opts.tmiinputs:
		print("TMI not supported yet.")
		quit()
	if opts.usetemporaryfolder:
		tempdir = opts.usetemporaryfolder[0]
		tempfolder = os.path.basename(tempdir)
		if not os.path.exists(tempdir):
			print("ERROR: %s does not exist" % tempdir)
			quit()
		if tempfolder == "":
			tempfolder = tempdir

		_, _, modtype = tempfolder.split('_')
		if modtype[-1] == "/":
			# should never happen...
			modtype = modtype[:-1]

		if os.path.exists("%s/nonzero.npy" % tempdir):
			data = np.load("%s/nonzero.npy" % tempdir)
			nonzero = np.load("%s/nonzero.npy" % tempdir)
		else:
			print("ERROR: %s/nonzero.py does not exist. Please re-import the data using {-si|-vi|-vil|-ti}.")
		print("Data loaded [%d intervals, %d subjects, %d voxels]" % (data.shape[0], data.shape[1], data.shape[2]))
		if modtype == 'volume':
			adjac = np.load("%s/adjac.npy" % tempdir)
			data_mask = np.load("%s/data_mask.npy" % tempdir)
			mask_index = np.load("%s/mask_index.npy" % tempdir)
			affine_mask = np.load("%s/affine_mask.npy" % tempdir)
			calcTFCE = CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjac)
			opts.volumetricinputs = True
		elif modtype == 'tmi':
			pass
		else:
			surface = modtype
			opts.surfaceinputfolder = True
			num_vertex_lh = np.load("%s/num_vertex_lh.npy" % tempdir)
			mask_lh = np.load("%s/mask_lh.npy" % tempdir)
			mask_rh = np.load("%s/mask_rh.npy" % tempdir)
			adjac_lh = np.load("%s/adjac_lh.npy" % tempdir)
			adjac_rh = np.load("%s/adjac_rh.npy" % tempdir)
			vdensity_lh = np.load("%s/vdensity_lh.npy" % tempdir)
			vdensity_rh = np.load("%s/vdensity_rh.npy" % tempdir)
			calcTFCE_lh = CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjac_lh)
			calcTFCE_rh = CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjac_rh)


	##### GLM ######
	if opts.generalizedlinearmodel:
		exog, varnames = load_vars(pdCSV, variables = opts.generalizedlinearmodel, exog = [], names = [])
		data = data[0] # There should only be one interval...


		if opts.covariates:
			covars, covarnames = load_vars(pdCSV, variables = opts.covariates, exog = [], names = [])
		else:
			covars = covarnames = []
		if opts.exogenousvariableinteraction:
			varnames, exog, covarnames, covars = load_interactions(opts.exogenousvariableinteraction, 
														varnames = varnames,
														exog = exog,
														covarnames = covarnames,
														covars = covars)
		if opts.covariates:
			dmy_covariates = np.concatenate(covars,1)
		else:
			dmy_covariates = None

		exog_shape = []
		for i in range(len(varnames)):
			exog_shape.append(exog[i].shape[1])

		Tvalues = Fmodel = Fvalues = None
		if opts.glmoutputstatistic[0] == 't':
			if opts.noreducedmodel:
				Tvalues = glm_typeI(data,
							exog,
							dmy_covariates=dmy_covariates,
							output_fvalues = False,
							output_tvalues = True,
							output_reduced_residuals = False,
							exog_names = varnames)
			else:
				Tvalues, data = glm_typeI(data,
							exog,
							dmy_covariates=dmy_covariates,
							output_fvalues = False,
							output_tvalues = True,
							output_reduced_residuals = True,
							exog_names = varnames)
		elif opts.glmoutputstatistic[0] == 'f':
			if opts.noreducedmodel:
				Fmodel, Fvalues = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_fvalues = True,
							output_tvalues = False,
							output_reduced_residuals = False)
			else:
				Fmodel, Fvalues, data = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_fvalues = True,
							output_tvalues = False,
							output_reduced_residuals = True,
							exog_names = varnames)
		else:
			if opts.noreducedmodel:
				Fmodel, Fvalues, Tvalues = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_fvalues = True,
							output_tvalues = True,
							output_reduced_residuals = False)
			else:
				Fmodel, Fvalues, Tvalues, data = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_fvalues = True,
							output_tvalues = True,
							output_reduced_residuals = True,
							exog_names = varnames)

		if opts.surfaceinputfolder:
			save_temporary_files('glm', modality_type = surface,
				all_vertex = all_vertex,
				num_vertex_lh = num_vertex_lh,
				mask_lh = mask_lh,
				mask_rh = mask_rh,
				adjac_lh = adjac_lh,
				adjac_rh = adjac_rh,
				vdensity_lh = vdensity_lh,
				vdensity_rh = vdensity_rh,
				exog_flat = np.concatenate(exog,1),
				exog_shape = exog_shape,
				dmy_covariates = dmy_covariates,
				optstfce = optstfce,
				varnames = varnames,
				gstat = opts.glmoutputstatistic[0],
				nonzero = nonzero.astype(np.float32, order = "C"),
				data = data.astype(np.float32, order = "C"))
		if opts.volumetricinputs or opts.volumetricinputlist:
			save_temporary_files('glm', modality_type = "volume",
				mask_index = mask_index,
				data_mask = data_mask,
				affine_mask = affine_mask,
				adjac = adjac,
				exog_flat = np.concatenate(exog,1),
				exog_shape = exog_shape,
				dmy_covariates = dmy_covariates,
				optstfce = optstfce,
				varnames = varnames,
				gstat = opts.glmoutputstatistic[0],
				nonzero = nonzero.astype(np.float32, order = "C"),
				data = data.astype(np.float32, order = "C"))

		if opts.surfaceinputfolder:
			outdir = "output_GLM_%s" % (surface)
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			os.chdir(outdir)

			if opts.covariates:
				np.savetxt("dmy_model.csv",
					stack_ones(np.column_stack((np.concatenate(exog,1), dmy_covariates))),
					delimiter=",")
			else:
				np.savetxt("dmy_model.csv",
					stack_ones(np.concatenate(exog,1)),
					delimiter=",")
			if Tvalues is not None:
				numcon = np.concatenate(exog,1).shape[1]
				for j in range(numcon):
					tnum=j+1
					write_vertStat_img('Tstat_con%d' % tnum, 
						Tvalues[tnum,:num_vertex_lh],
						outdata_mask_lh,
						affine_mask_lh,
						surface,
						'lh',
						mask_lh,
						calcTFCE_lh,
						mask_lh.shape[0],
						vdensity_lh)
					write_vertStat_img('Tstat_con%d' % tnum,
						Tvalues[tnum,num_vertex_lh:],
						outdata_mask_rh,
						affine_mask_rh,
						surface,
						'rh',
						mask_rh,
						calcTFCE_rh,
						mask_rh.shape[0],
						vdensity_rh)
					write_vertStat_img('negTstat_con%d' % tnum,
						(Tvalues[tnum,:num_vertex_lh]*-1),
						outdata_mask_lh,
						affine_mask_lh,
						surface,
						'lh',
						mask_lh,
						calcTFCE_lh,
						mask_lh.shape[0],
						vdensity_lh)
					write_vertStat_img('negTstat_con%d' % tnum,
						(Tvalues[tnum,num_vertex_lh:]*-1),
						outdata_mask_rh,
						affine_mask_rh,
						surface,
						'rh',
						mask_rh,
						calcTFCE_rh,
						mask_rh.shape[0],
						vdensity_rh)
			if Fvalues is not None:
				write_vertStat_img('Fmodel', 
					Fmodel[:num_vertex_lh],
					outdata_mask_lh,
					affine_mask_lh,
					surface,
					'lh',
					mask_lh,
					calcTFCE_lh,
					mask_lh.shape[0],
					vdensity_lh)
				write_vertStat_img('Fmodel',
					Fmodel[num_vertex_lh:],
					outdata_mask_rh,
					affine_mask_rh,
					surface,
					'rh',
					mask_rh,
					calcTFCE_rh,
					mask_rh.shape[0],
					vdensity_rh)
				for j in range(len(exog)):
					conname = 'Fstat_%s' % varnames[j]
					write_vertStat_img(conname, 
						Fvalues[j,:num_vertex_lh],
						outdata_mask_lh,
						affine_mask_lh,
						surface,
						'lh',
						mask_lh,
						calcTFCE_lh,
						mask_lh.shape[0],
						vdensity_lh)
					write_vertStat_img(conname,
						Fvalues[j,num_vertex_lh:],
						outdata_mask_rh,
						affine_mask_rh,
						surface,
						'rh',
						mask_rh,
						calcTFCE_rh,
						mask_rh.shape[0],
						vdensity_rh)

		if opts.volumetricinputs or opts.volumetricinputlist:
			outdir = "output_GLM_volume"
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			os.chdir(outdir)
			if opts.covariates:
				np.savetxt("dmy_model.csv",
					stack_ones(np.column_stack((np.concatenate(exog,1), dmy_covariates))),
					delimiter=",")
			else:
				np.savetxt("dmy_model.csv",
					stack_ones(np.concatenate(exog,1)),
					delimiter=",")
			if Tvalues is not None:
				numcon = np.concatenate(exog,1).shape[1]
				for j in range(numcon):
					tnum=j+1
					write_voxelStat_img('Tstat_con%d' % tnum, 
						Tvalues[tnum,:],
						data_mask,
						mask_index,
						affine_mask,
						calcTFCE,
						imgext)
					write_voxelStat_img('negTstat_con%d' % tnum, 
						-Tvalues[tnum,:],
						data_mask,
						mask_index,
						affine_mask,
						calcTFCE,
						imgext)
			if Fvalues is not None:
				write_voxelStat_img('Fmodel', 
					Fmodel,
					data_mask,
					mask_index,
					affine_mask,
					calcTFCE,
					imgext)
				for j in range(len(exog)):
					conname = 'Fstat_%s' % varnames[j]
					write_voxelStat_img(conname,
						Fvalues[j,:],
						data_mask,
						mask_index,
						affine_mask,
						calcTFCE,
						imgext)

	if opts.mediation:
		medtype = opts.mediation[0]
		factors = opts.mediation[1:]
		data = data[0] # There should only be one interval...

		if factors[1] == 'c':
			dmy_leftvar = dummy_code(np.array(pdCSV[factors[0]]), iscontinous = True)
			print("Coding %s as continous variable" % factors[0])
		else:
			dmy_leftvar = dummy_code(np.array(pdCSV[factors[0]]), iscontinous = False)
			print("Coding %s as discrete variable" % factors[0])
		if factors[3] == 'c':
			dmy_rightvar = dummy_code(np.array(pdCSV[factors[2]]), iscontinous = True)
			print("Coding %s as continous variable" % factors[2])
		else:
			dmy_rightvar = dummy_code(np.array(pdCSV[factors[2]]), iscontinous = False)
			print("Coding %s as discrete variable" % factors[2])

		if opts.covariates:
			covars, covarnames = load_vars(pdCSV, variables = opts.covariates, exog = [], names = [])
		else:
			covars = covarnames = []

		if opts.exogenousvariableinteraction:
			_, _, covarnames, covars = load_interactions(opts.exogenousvariableinteraction, 
														varnames = [],
														exog = [],
														covarnames = covarnames,
														covars = covars)

		if opts.covariates:
			dmy_covariates = np.concatenate(covars,1)
		else:
			dmy_covariates = None

		if medtype == 'I':
			EXOG_A = []
			EXOG_A.append(dmy_leftvar)
			EXOG_B = []
			EXOG_B.append(dmy_leftvar)
			EXOG_B.append(dmy_rightvar)

			Tvalues_A = glm_typeI(data,
						EXOG_A,
						dmy_covariates = dmy_covariates,
						output_fvalues = False,
						output_tvalues = True,
						output_reduced_residuals = False)[1]
			Tvalues_B = glm_typeI(data,
						EXOG_B,
						dmy_covariates=dmy_covariates,
						output_fvalues = False,
						output_tvalues = True,
						output_reduced_residuals = False)[1]
		elif medtype == 'M':
			EXOG_A = []
			EXOG_A.append(dmy_leftvar)
			EXOG_B = []
			EXOG_B.append(dmy_rightvar)
			EXOG_B.append(dmy_leftvar)

			Tvalues_A = glm_typeI(data,
						EXOG_A,
						dmy_covariates = dmy_covariates,
						output_fvalues = False,
						output_tvalues = True,
						output_reduced_residuals = False)[1]
			Tvalues_B = glm_typeI(data,
						EXOG_B,
						dmy_covariates=dmy_covariates,
						output_fvalues = False,
						output_tvalues = True,
						output_reduced_residuals = False)[1]
		elif medtype == 'Y':
			EXOG_A = []
			EXOG_A.append(dmy_leftvar)
			EXOG_B = []
			EXOG_B.append(dmy_rightvar)
			EXOG_B.append(dmy_leftvar)

			Tvalues_A = glm_typeI(dmy_rightvar,
						EXOG_A,
						dmy_covariates=dmy_covariates,
						output_fvalues = False,
						output_tvalues = True,
						output_reduced_residuals = False)[1]

			Tvalues_B = glm_typeI(data,
						EXOG_B,
						dmy_covariates=dmy_covariates,
						output_fvalues = False,
						output_tvalues = True,
						output_reduced_residuals = False)[1]
		else:
			print("ERROR: Invalid mediation type: %s" % medtype)
			quit()

		SobelZ  = calc_indirect(Tvalues_A, Tvalues_B, alg = "aroian")

		if opts.surfaceinputfolder:
			save_temporary_files('mediation', modality_type = surface,
				all_vertex = all_vertex,
				num_vertex_lh = num_vertex_lh,
				mask_lh = mask_lh,
				mask_rh = mask_rh,
				adjac_lh = adjac_lh,
				adjac_rh = adjac_rh,
				vdensity_lh = vdensity_lh,
				vdensity_rh = vdensity_rh,
				dmy_leftvar = dmy_leftvar,
				dmy_rightvar = dmy_rightvar,
				dmy_covariates = dmy_covariates,
				optstfce = optstfce,
				medtype = medtype,
				nonzero = nonzero.astype(np.float32, order = "C"),
				data = data.astype(np.float32, order = "C"))

			outdir = "output_mediation_%s" % (surface)
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			os.chdir(outdir)

			write_vertStat_img('Zstat_%s' % medtype,
				SobelZ[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Zstat_%s' % medtype,
				SobelZ[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)

		if opts.volumetricinputs or opts.volumetricinputlist:
			save_temporary_files('mediation', modality_type = "volume",
				mask_index = mask_index,
				data_mask = data_mask,
				affine_mask = affine_mask,
				adjac = adjac,
				dmy_leftvar = dmy_leftvar,
				dmy_rightvar = dmy_rightvar,
				dmy_covariates = dmy_covariates,
				optstfce = optstfce,
				medtype = medtype,
				nonzero = nonzero.astype(np.float32, order = "C"),
				data = data.astype(np.float32, order = "C"))

			outdir = "output_mediation_volume"
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			os.chdir(outdir)

			write_voxelStat_img('Zstat_%s' % medtype,
				SobelZ,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)


	##### RM ANCOVA (one between subject, one within subject) ######
	if opts.onebetweenssubjectfactor:
		factors = opts.onebetweenssubjectfactor
		subjects = opts.subjectidcolumns[0]

		if factors[1] == 'c':
			dmy_factor1 = dummy_code(np.array(pdCSV[factors[0]]), iscontinous = True)
			print("Coding %s as continous variable" % factors[0])
		else:
			dmy_factor1 = dummy_code(np.array(pdCSV[factors[0]]), iscontinous = False)
			print("Coding %s as discrete variable" % factors[0])
		dmy_subjects = dummy_code(np.array(pdCSV[subjects]), demean = False)

		if opts.covariates:
			covars, covarnames = load_vars(pdCSV, variables = opts.covariates, exog = [], names = [])
		else:
			covars = covarnames = []

		if opts.exogenousvariableinteraction:
			_, _, covarnames, covars = load_interactions(opts.exogenousvariableinteraction, 
																	varnames = [],
																	exog = [],
																	covarnames = covarnames,
																	covars = covars)

		if opts.covariates:
			dmy_covariates = np.concatenate(covars,1)
		else:
			dmy_covariates = None

		if opts.noreducedmodel:
			dformat = np.array(['short'])
			F_a, F_s, F_sa = reg_rm_ancova_one_bs_factor(data, 
									dmy_factor1,
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False)
		else:
			dformat = np.array(['long'])
			F_a, F_s, F_sa, data = reg_rm_ancova_one_bs_factor(data, 
									dmy_factor1,
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False,
									output_reduced_residuals = True)

		if opts.surfaceinputfolder:
			save_temporary_files('rmancova_one', modality_type = surface,
				all_vertex = all_vertex,
				num_vertex_lh = num_vertex_lh,
				mask_lh = mask_lh,
				mask_rh = mask_rh,
				adjac_lh = adjac_lh,
				adjac_rh = adjac_rh,
				vdensity_lh = vdensity_lh,
				vdensity_rh = vdensity_rh,
				dmy_factor1 = dmy_factor1,
				dmy_subjects = dmy_subjects,
				dmy_covariates = dmy_covariates,
				optstfce = optstfce,
				factors = factors,
				dformat = dformat,
				nonzero = nonzero.astype(np.float32, order = "C"),
				data = data.astype(np.float32, order = "C"))
		if opts.volumetricinputs or opts.volumetricinputlist:
			save_temporary_files('rmancova_one', modality_type = "volume",
				mask_index = mask_index,
				data_mask = data_mask,
				affine_mask = affine_mask,
				adjac = adjac,
				dmy_factor1 = dmy_factor1,
				dmy_subjects = dmy_subjects,
				dmy_covariates = dmy_covariates,
				optstfce = optstfce,
				factors = factors,
				dformat = dformat,
				nonzero = nonzero.astype(np.float32, order = "C"),
				data = data.astype(np.float32, order = "C"))

		if opts.surfaceinputfolder:
			outdir = "output_rmANCOVA1BS_%s" % (surface)
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			os.chdir(outdir)

			write_vertStat_img('Fstat_%s' % factors[0], 
				F_a[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_%s' % factors[0], 
				F_a[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)
			write_vertStat_img('Fstat_time',
				F_s[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_time',
				F_s[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)
			write_vertStat_img('Fstat_%s.X.time' % factors[0], 
				F_sa[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_%s.X.time' % factors[0], 
				F_sa[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)

		if opts.volumetricinputs or opts.volumetricinputlist:
			outdir = "output_rmANCOVA1BS_volume"
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			os.chdir(outdir)
			write_voxelStat_img('Fstat_%s' % factors[0], 
				F_a,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)
			write_voxelStat_img('Fstat_time', 
				F_s,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)
			write_voxelStat_img('Fstat_%s.X.time' % factors[0], 
				F_sa,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)


	##### RM ANCOVA (two between subject, one within subject) ######
	if opts.twobetweenssubjectfactor:
		factors = opts.twobetweenssubjectfactor
		subjects = opts.subjectidcolumns[0]

		if factors[1] == 'c':
			dmy_factor1 = dummy_code(np.array(pdCSV[factors[0]]), iscontinous = True)
			print("Coding %s as continous variable" % factors[0])
		else:
			dmy_factor1 = dummy_code(np.array(pdCSV[factors[0]]), iscontinous = False)
			print("Coding %s as discrete variable" % factors[0])
		if factors[3] == 'c':
			dmy_factor2 = dummy_code(np.array(pdCSV[factors[2]]), iscontinous = True)
			print("Coding %s as continous variable" % factors[2])
		else:
			dmy_factor2 = dummy_code(np.array(pdCSV[factors[2]]), iscontinous = False)
			print("Coding %s as discrete variable" % factors[2])
		dmy_subjects = dummy_code(np.array(pdCSV[subjects]), demean = False)


		if opts.covariates:
			covars, covarnames = load_vars(pdCSV, variables = opts.covariates, exog = [], names = [])
		else:
			covars = covarnames = []

		if opts.exogenousvariableinteraction:

			_, _, covarnames, covars = load_interactions(opts.exogenousvariableinteraction, 
																	varnames = [],
																	exog = [],
																	covarnames = covarnames,
																	covars = covars)

		if opts.covariates:
			dmy_covariates = np.concatenate(covars,1)
		else:
			dmy_covariates = None

		if opts.noreducedmodel:
			dformat = np.array(['short'])
			F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab = reg_rm_ancova_two_bs_factor(data, 
									dmy_factor1,
									dmy_factor2, 
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False)
		else:
			dformat = np.array(['long'])
			F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab, data = reg_rm_ancova_two_bs_factor(data, 
									dmy_factor1,
									dmy_factor2, 
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False,
									output_reduced_residuals = True)

		if opts.surfaceinputfolder:
			save_temporary_files('rmancova_one', modality_type = surface,
				all_vertex = all_vertex,
				num_vertex_lh = num_vertex_lh,
				mask_lh = mask_lh,
				mask_rh = mask_rh,
				adjac_lh = adjac_lh,
				adjac_rh = adjac_rh,
				vdensity_lh = vdensity_lh,
				vdensity_rh = vdensity_rh,
				dmy_factor1 = dmy_factor1,
				dmy_factor2 = dmy_factor2,
				dmy_subjects = dmy_subjects,
				dmy_covariates = dmy_covariates,
				optstfce = optstfce,
				factors = factors,
				dformat = dformat,
				nonzero = nonzero.astype(np.float32, order = "C"),
				data = data.astype(np.float32, order = "C"))
		if opts.volumetricinputs or opts.volumetricinputlist:
			save_temporary_files('rmancova_one', modality_type = "volume",
				mask_index = mask_index,
				data_mask = data_mask,
				affine_mask = affine_mask,
				adjac = adjac,
				dmy_factor1 = dmy_factor1,
				dmy_factor2 = dmy_factor2,
				dmy_subjects = dmy_subjects,
				dmy_covariates = dmy_covariates,
				optstfce = optstfce,
				factors = factors,
				dformat = dformat,
				nonzero = nonzero.astype(np.float32, order = "C"),
				data = data.astype(np.float32, order = "C"))

		if opts.surfaceinputfolder:
			outdir = "output_rmANCOVA2BS_%s" % (surface)
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			os.chdir(outdir)

			# Between Subjects
			write_vertStat_img('Fstat_%s' % factors[0], 
				F_a[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_%s' % factors[0], 
				F_a[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)
			write_vertStat_img('Fstat_%s' % factors[2], 
				F_b[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_%s' % factors[2], 
				F_b[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)
			write_vertStat_img('Fstat_%s.X.%s' % (factors[0],factors[2]), 
				F_ab[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_%s.X.%s' % (factors[0],factors[2]), 
				F_ab[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)

			# Within Subjects
			write_vertStat_img('Fstat_time', 
				F_s[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_time',  
				F_s[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)

			write_vertStat_img('Fstat_%s.X.time' % factors[0], 
				F_sa[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_%s.X.time' % factors[0], 
				F_sa[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)
			write_vertStat_img('Fstat_%s.X.time' % factors[2], 
				F_sb[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_%s.X.time' % factors[2], 
				F_sb[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)
			write_vertStat_img('Fstat_%s.X.%s.X.time' % (factors[0], factors[2]), 
				F_sab[:num_vertex_lh],
				outdata_mask_lh,
				affine_mask_lh,
				surface,
				'lh',
				mask_lh,
				calcTFCE_lh,
				mask_lh.shape[0],
				vdensity_lh)
			write_vertStat_img('Fstat_%s.X.%s.X.time' % (factors[0], factors[2]), 
				F_sab[num_vertex_lh:],
				outdata_mask_rh,
				affine_mask_rh,
				surface,
				'rh',
				mask_rh,
				calcTFCE_rh,
				mask_rh.shape[0],
				vdensity_rh)

		if opts.volumetricinputs or opts.volumetricinputlist:

			outdir = "output_rmANCOVA2BS_volume"
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			os.chdir(outdir)

			write_voxelStat_img('Fstat_%s' % factors[0], 
				F_a,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)
			write_voxelStat_img('Fstat_%s' % factors[2], 
				F_b,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)
			write_voxelStat_img('Fstat_%s.X.%s' % (factors[0],factors[2]),
				F_ab,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)
			write_voxelStat_img('Fstat_time',
				F_s,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)
			write_voxelStat_img('Fstat_%s.X.time' % factors[0],
				F_sa,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)
			write_voxelStat_img('Fstat_%s.X.time' % factors[2],
				F_sb,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)
			write_voxelStat_img('Fstat_%s.X.%s.X.time' % (factors[0], factors[2]),
				F_sb,
				data_mask,
				mask_index,
				affine_mask,
				calcTFCE,
				imgext)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
