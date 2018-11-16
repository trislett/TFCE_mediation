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
from tfce_mediation.pyfunc import write_vertStat_img, create_adjac_vertex, reg_rm_ancova_one_bs_factor, reg_rm_ancova_two_bs_factor, glm_typeI, dummy_code, column_product, stack_ones

def check_columns(pdData, folders, surface, FWHM):
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
		if interaction_vars[0] in varnames:
			for i, scale_var in enumerate(interaction_vars):
				if i == 0:
					int_temp = exog[varnames.index(interaction_vars[i])]
				else:
					int_temp = column_product(int_temp, exog[varnames.index(interaction_vars[i])])
			exog.append(int_temp)
			varnames.append(int_terms)
		elif interaction_vars[0] in covarnames:
			for i, scale_var in enumerate(interaction_vars):
				if i == 0:
					int_temp = covars[covarnames.index(interaction_vars[i])]
				else:
					int_temp = column_product(int_temp, covars[covarnames.index(interaction_vars[i])])
			covars.append(int_temp)
			covarnames.append(int_terms)
		else:
			print("Error: interaction variables must be contained in -glm or -c")
	return (varnames, exog, covarnames, covars)


DESCRIPTION = "Vertex-wise GLMs include within-subject interactions with TFCE."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):

	ap.add_argument("-i", "--inputcsv", 
		nargs=1, 
		help="Input folder containing each surface interval. e.g., -sa BL POST FU6 FU12 [-si {folder1} .. {foldern}]", 
		metavar=('*.csv'),
		required=True)
	ap.add_argument("-sc", "--subjectidcolumns",
		help="Select the subject id column for merging *.csv files. Default: %(default)s)",
		nargs=1,
		default=['SubjID'])
	ap.add_argument("-on", "--outputcolumnnames", 
		help="Outputs the input CSV column names, and check the column length and number of images.", 
		action='store_true')

	ap.add_argument("-si", "--surfaceinputfolder", 
		nargs='+', 
		help="Input folder containing each surface interval. -si {folder1} .. {foldern}", 
		metavar=('PATH/TO/DIR'),
		required=True)
	ap.add_argument("-s", "--surface",  
		nargs=1, 
		metavar=('{area|thickness}'), 
		required=True)
	ap.add_argument("-f", "--fwhm", 
		help="Specific all surface file with different smoothing. Default is 03B (recommended)" , 
		nargs=1, 
		default=['03B'], 
		metavar=('??B'))

	stat = ap.add_mutually_exclusive_group(required=False)
	stat.add_argument("-glm","--generalizedlinearmodel",
		nargs='+',
		help="Generalized linear model that uses type I sum of squares. Each exogenous variable must be specified as either discrete (d) or continous (c). Output metrics can be specified by -gs as either F-statics, T-statistics, or all. Interactions can be specified using -ei. e.g., -glm sex d age c genotype d. [-glm {exogenous_1} {type_1} ... {exogenous_k} {type_k}]",
		metavar=('exog1', '{d|c}'))
	stat.add_argument("-ofa","--onebetweenssubjectfactor",
		nargs=2,
		help="ANCOVA with one between-subject (fixed) factor and neuroimage as the within-subject (random) factor (mixed-effect model). AKA, the repeated-measure ANCOVA. Covariates may also be included. Interactions will be coded automatically (Factor1*Time). ANOVA uses type I sum of squares (order matters). The between-subject factor must be specified as either discrete (d) or continous (c). Additional interactions among the covariates can be specified using -ei. e.g., -ofa genotype d. [-ofa {factor} {type}]",
		metavar=('exog1', '{d|c}'))
	stat.add_argument("-tfa","--twobetweenssubjectfactor",
		nargs=4,
		help="ANCOVA with two between-subject (fixed) factors and neuroimage as the within-subject (random) factor (mixed-effect model). AKA, the repeated-measure ANCOVA. Covariates may also be included. Interactions will be coded automatically (F1*F2, F1*Time,F2xTime, F1*F2*Time). ANOVA uses type I sum of squares (order matters). The between-subject factor must be specified as either discrete (d) or continous (c). Additional interactions among the covariates can be specified using -ei. e.g., -tfa sex d genotype d. [-tfa {factor1} {type1} {factor2} {type2}]",
		metavar=('exog1', '{d|c}', 'exog2', '{d|c}'))

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
	mask.add_argument("--fmri", 
		help="Masking threshold for fMRI surfaces. Default is 0.1 (i.e., mask regions with values less than -0.1 and greater than 0.1)",
		const=0.1, 
		type=float, 
		nargs='?')
	mask.add_argument("-m","--fsmask", 
		help="Create masking array based on fsaverage label. Default is cortex", 
		const='cortex', 
		nargs='?')
	mask.add_argument("-l","--label", 
		help="Load label as masking array for lh and rh, respectively.", 
		nargs=2, 
		metavar=('lh.*.label','rh.*.label'))
	mask.add_argument("-b","--binmask", 
		help="Load binary mask surface for lh and rh, respectively.", 
		nargs=2, 
		metavar=('lh.*.mgh','rh.*.mgh'))

	adjac = ap.add_mutually_exclusive_group(required=False)
	adjac.add_argument("-d", "--dist", 
		help="Load supplied adjacency sets geodesic distance in mm. Default is 3 (recommended).", 
		choices = [1,2,3], 
		type=int,  
		nargs=1, 
		default=[3])
	adjac.add_argument("-a", "--adjfiles", help="Load custom adjacency set for each hemisphere.", 
		nargs=2, 
		metavar=('*.npy', '*.npy'))
	adjac.add_argument("-t", "--triangularmesh", 
		help="Create adjacency based on triangular mesh without specifying distance.",
		action='store_true')
	ap.add_argument("--inputsurfs", 
		help="Load surfaces for triangular mesh tfce. --triangularmesh option must be used.",
		nargs=2,
		metavar=('lh.*.srf', 'rh.*.srf'))
	ap.add_argument("--tfce", 
		help="TFCE settings. H (i.e., height raised to power H), E (i.e., extent raised to power E). Default: %(default)s). H=2, E=2/3 is the point at which the cummulative density function is approximately Gaussian distributed.", 
		nargs=2, 
		default=[2,0.67], 
		metavar=('H', 'E'))
	ap.add_argument("--noweight", 
		help="Do not weight each vertex for density of vertices within the specified geodesic distance.", 
		action="store_true")
	ap.add_argument("--noreducedmodel", 
		help="Do not use a reduced model for permutation testing (i.e., the residuals after controlling for between-subject effects).",
		action="store_true")
	return ap

def run(opts):

	scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	surface = opts.surface[0]
	FWHM = opts.fwhm[0]
	CSV = opts.inputcsv[0]
	pdCSV = pd.read_csv(CSV, delimiter=',', index_col=None)
	folders = opts.surfaceinputfolder

	# output column/variable names.
	if opts.outputcolumnnames:
		check_columns(pdCSV, folders, surface, FWHM)
		quit()
	else:
		# secondary required variables
		if opts.generalizedlinearmodel:
			pass
		elif opts.onebetweenssubjectfactor:
			pass
		elif opts.twobetweenssubjectfactor:
			pass
		else:
			print("Error: please specify statistical model {-glm | -of | -tw}")
			quit()


	#load surface data
	data = []

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
			num_vertex_rh = data_rh[mask_rh].shape[0]
			all_vertex = data_full_lh.shape[0]

		else:
			data_lh = np.squeeze(nib.freesurfer.mghformat.load("%s/lh.all.%s.%s.mgh" % (sfolder, surface,FWHM)).get_data())
			data_rh = np.squeeze(nib.freesurfer.mghformat.load("%s/rh.all.%s.%s.mgh" % (sfolder, surface,FWHM)).get_data())
			data.append(np.hstack((data_lh[mask_lh].T,data_rh[mask_rh].T)))
			data_lh = data_rh = []
	data = np.array(data)

	#TFCE
	if opts.triangularmesh:
		print("Creating adjacency set")
		if opts.inputsurfs:
			# 3 Neighbour vertex connectity
			v_lh, faces_lh = nib.freesurfer.read_geometry(opts.inputsurfs[0])
			v_rh, faces_rh = nib.freesurfer.read_geometry(opts.inputsurfs[1])
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
	elif opts.dist:
		print("Loading prior adjacency set for %s mm" % opts.dist[0])
		adjac_lh = np.load("%s/adjacency_sets/lh_adjacency_dist_%s.0_mm.npy" % (scriptwd,str(opts.dist[0])))
		adjac_rh = np.load("%s/adjacency_sets/rh_adjacency_dist_%s.0_mm.npy" % (scriptwd,str(opts.dist[0])))
	else:
		print("Error")
	if opts.noweight or opts.triangularmesh:
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


	exog = varnames = []
	##### GLM ######
	if opts.generalizedlinearmodel:
		exog, varnames = load_vars(pdCSV, variables = opts.generalizedlinearmodel, exog = [], names = [])
		data = data[0] # There should only be one interval...

		exog_shape = []
		for i in range(len(varnames)):
			exog_shape.append(exog[i].shape[1])

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

		#save variables
		if not os.path.exists("tmp_tmGLM_%s" % (surface)):
			os.mkdir("tmp_tmGLM_%s" % (surface))
		np.save("tmp_tmGLM_%s/all_vertex" % (surface),all_vertex)
		np.save("tmp_tmGLM_%s/num_vertex_lh" % (surface),num_vertex_lh)
		np.save("tmp_tmGLM_%s/num_vertex_rh" % (surface),num_vertex_rh)
		np.save("tmp_tmGLM_%s/mask_lh" % (surface),mask_lh)
		np.save("tmp_tmGLM_%s/mask_rh" % (surface),mask_rh)
		np.save("tmp_tmGLM_%s/affine_mask_lh" % (surface),affine_mask_lh)
		np.save("tmp_tmGLM_%s/affine_mask_rh" % (surface),affine_mask_rh)
		np.save("tmp_tmGLM_%s/adjac_lh" % (surface),adjac_lh)
		np.save("tmp_tmGLM_%s/adjac_rh" % (surface),adjac_rh)
		np.save("tmp_tmGLM_%s/exog_flat" % (surface),np.concatenate(exog,1))
		np.save("tmp_tmGLM_%s/exog_shape" % (surface),exog_shape)
		np.save("tmp_tmGLM_%s/dmy_covariates" % (surface),dmy_covariates)
		np.save("tmp_tmGLM_%s/optstfce" % (surface), opts.tfce)
		np.save("tmp_tmGLM_%s/vdensity_lh" % (surface), vdensity_lh)
		np.save("tmp_tmGLM_%s/vdensity_rh" % (surface), vdensity_rh)
		np.save("tmp_tmGLM_%s/varnames" % (surface), varnames)
		np.save("tmp_tmGLM_%s/gstat" % (surface), opts.glmoutputstatistic[0])

		Tvalues = Fmodel = Fvalues = None
		if opts.glmoutputstatistic[0] == 't':
			if opts.noreducedmodel:
				Tvalues = glm_typeI(data,
							exog,
							dmy_covariates=dmy_covariates,
							output_fvalues = False,
							output_tvalues = True,
							output_reduced_residuals = False)
			else:
				Tvalues, data = glm_typeI(data,
							exog,
							dmy_covariates=dmy_covariates,
							output_fvalues = False,
							output_tvalues = True,
							output_reduced_residuals = True)
		elif opts.glmoutputstatistic[0] == 'f':
			if opts.noreducedmodel:
				Fmodel, Fvalues = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_reduced_residuals = False)
			else:
				Fmodel, Fvalues, data = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_reduced_residuals = True)
		else:
			if opts.noreducedmodel:
				Fmodel, Fvalues, Tvalues = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_tvalues =True,
							output_reduced_residuals = False)
			else:
				Fmodel, Fvalues, Tvalues, data = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_tvalues =True,
							output_reduced_residuals = True)
		# now saves the reduced model.
		np.save("tmp_tmGLM_%s/data" % (surface),data.astype(np.float32, order = "C"))

		#write TFCE images
		if not os.path.exists("outputGLM_%s" % (surface)):
			os.mkdir("outputGLM_%s" % (surface))
		os.chdir("outputGLM_%s" % (surface))
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
				write_vertStat_img('tstat_con%d' % tnum, 
					Tvalues[tnum,:num_vertex_lh],
					outdata_mask_lh,
					affine_mask_lh,
					surface,
					'lh',
					mask_lh,
					calcTFCE_lh,
					mask_lh.shape[0],
					vdensity_lh)
				write_vertStat_img('tstat_con%d' % tnum,
					Tvalues[tnum,num_vertex_lh:],
					outdata_mask_rh,
					affine_mask_rh,
					surface,
					'rh',
					mask_rh,
					calcTFCE_rh,
					mask_rh.shape[0],
					vdensity_rh)
				write_vertStat_img('negtstat_con%d' % tnum,
					(Tvalues[tnum,:num_vertex_lh]*-1),
					outdata_mask_lh,
					affine_mask_lh,
					surface,
					'lh',
					mask_lh,
					calcTFCE_lh,
					mask_lh.shape[0],
					vdensity_lh)
				write_vertStat_img('negtstat_con%d' % tnum,
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
		exog = []
		varnames = []

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


		#save variables
		if not os.path.exists("tmp_tmANCOVA1BS_%s" % (surface)):
			os.mkdir("tmp_tmANCOVA1BS_%s" % (surface))
		np.save("tmp_tmANCOVA1BS_%s/all_vertex" % (surface),all_vertex)
		np.save("tmp_tmANCOVA1BS_%s/num_vertex_lh" % (surface),num_vertex_lh)
		np.save("tmp_tmANCOVA1BS_%s/num_vertex_rh" % (surface),num_vertex_rh)
		np.save("tmp_tmANCOVA1BS_%s/mask_lh" % (surface),mask_lh)
		np.save("tmp_tmANCOVA1BS_%s/mask_rh" % (surface),mask_rh)
		np.save("tmp_tmANCOVA1BS_%s/affine_mask_lh" % (surface),affine_mask_lh)
		np.save("tmp_tmANCOVA1BS_%s/affine_mask_rh" % (surface),affine_mask_rh)
		np.save("tmp_tmANCOVA1BS_%s/adjac_lh" % (surface),adjac_lh)
		np.save("tmp_tmANCOVA1BS_%s/adjac_rh" % (surface),adjac_rh)
		np.save("tmp_tmANCOVA1BS_%s/dmy_factor1" % (surface),dmy_factor1)
		np.save("tmp_tmANCOVA1BS_%s/dmy_covariates" % (surface),dmy_covariates)
		np.save("tmp_tmANCOVA1BS_%s/dmy_subjects" % (surface),dmy_subjects)
		np.save("tmp_tmANCOVA1BS_%s/optstfce" % (surface), opts.tfce)
		np.save("tmp_tmANCOVA1BS_%s/vdensity_lh" % (surface), vdensity_lh)
		np.save("tmp_tmANCOVA1BS_%s/vdensity_rh" % (surface), vdensity_rh)
		np.save("tmp_tmANCOVA1BS_%s/factors" % (surface), factors)

		if opts.noreducedmodel:
			np.save("tmp_tmANCOVA1BS_%s/dformat" % (surface),np.array(['short']))
			F_a, F_s, F_sa = reg_rm_ancova_one_bs_factor(data, 
									dmy_factor1,
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False)
		else:
			np.save("tmp_tmANCOVA1BS_%s/dformat" % (surface),np.array(['long']))
			F_a, F_s, F_sa, data = reg_rm_ancova_one_bs_factor(data, 
									dmy_factor1,
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False,
									output_reduced_residuals = True)
		np.save("tmp_tmANCOVA1BS_%s/data" % (surface),data.astype(np.float32, order = "C"))


		if not os.path.exists("outputANCOVA1BS_%s" % (surface)):
			os.mkdir("outputANCOVA1BS_%s" % (surface))
		os.chdir("outputANCOVA1BS_%s" % (surface))

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


		#save variables
		if not os.path.exists("tmp_tmANCOVA2BS_%s" % (surface)):
			os.mkdir("tmp_tmANCOVA2BS_%s" % (surface))
		np.save("tmp_tmANCOVA2BS_%s/all_vertex" % (surface),all_vertex)
		np.save("tmp_tmANCOVA2BS_%s/num_vertex_lh" % (surface),num_vertex_lh)
		np.save("tmp_tmANCOVA2BS_%s/num_vertex_rh" % (surface),num_vertex_rh)
		np.save("tmp_tmANCOVA2BS_%s/mask_lh" % (surface),mask_lh)
		np.save("tmp_tmANCOVA2BS_%s/mask_rh" % (surface),mask_rh)
		np.save("tmp_tmANCOVA2BS_%s/affine_mask_lh" % (surface),affine_mask_lh)
		np.save("tmp_tmANCOVA2BS_%s/affine_mask_rh" % (surface),affine_mask_rh)
		np.save("tmp_tmANCOVA2BS_%s/adjac_lh" % (surface),adjac_lh)
		np.save("tmp_tmANCOVA2BS_%s/adjac_rh" % (surface),adjac_rh)
		np.save("tmp_tmANCOVA2BS_%s/data" % (surface),data.astype(np.float32, order = "C"))
		np.save("tmp_tmANCOVA2BS_%s/dmy_factor1" % (surface),dmy_factor1)
		np.save("tmp_tmANCOVA2BS_%s/dmy_factor2" % (surface),dmy_factor2)
		np.save("tmp_tmANCOVA2BS_%s/dmy_covariates" % (surface),dmy_covariates)
		np.save("tmp_tmANCOVA2BS_%s/dmy_subjects" % (surface),dmy_subjects)
		np.save("tmp_tmANCOVA2BS_%s/optstfce" % (surface), opts.tfce)
		np.save("tmp_tmANCOVA2BS_%s/vdensity_lh" % (surface), vdensity_lh)
		np.save("tmp_tmANCOVA2BS_%s/vdensity_rh" % (surface), vdensity_rh)
		np.save("tmp_tmANCOVA2BS_%s/factors" % (surface), factors)

		if opts.noreducedmodel:
			np.save("tmp_tmANCOVA2BS_%s/dformat" % (surface),np.array(['short']))
			F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab = reg_rm_ancova_two_bs_factor(data, 
									dmy_factor1,
									dmy_factor2, 
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False)
		else:
			np.save("tmp_tmANCOVA2BS_%s/dformat" % (surface),np.array(['long']))
			F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab, data = reg_rm_ancova_two_bs_factor(data, 
									dmy_factor1,
									dmy_factor2, 
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False,
									output_reduced_residuals = True)
		np.save("tmp_tmANCOVA2BS_%s/data" % (surface),data.astype(np.float32, order = "C"))

		if not os.path.exists("outputANCOVA2BS_%s" % (surface)):
			os.mkdir("outputANCOVA2BS_%s" % (surface))
		os.chdir("outputANCOVA2BS_%s" % (surface))

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

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
