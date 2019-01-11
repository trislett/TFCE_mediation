#!/usr/bin/env python

#    Vertex-wise stastical models including mediation, within-subject designs, and GLMs with TFCE
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
import argparse as ap
from time import time

from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.pyfunc import reg_rm_ancova_one_bs_factor, reg_rm_ancova_two_bs_factor, glm_typeI, glm_cosinor, write_perm_maxTFCE_vertex, write_perm_maxTFCE_voxel, calc_indirect, check_blocks, rand_blocks

DESCRIPTION = "Vertex-wise stastical models including mediation, within-subject designs, and GLMs with TFCE."
start_time = time()

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-r", "--range", 
		nargs = 2, 
		type = int, 
		help = "permutation [start] [stop]", 
		metavar = ('INT','INT'), 
		required = True)
	modality = ap.add_mutually_exclusive_group(required=True)
	modality.add_argument("-s", "--surface", 
		nargs = 1, 
		help = "Randomise surface analysis. -s {surface}", 
		metavar = ('STR'))
	modality.add_argument("-v", "--voxel",
		action = 'store_true',
		help = "Randomise volumetric analysis")
	modality.add_argument("-t", "--tmi", 
		action = 'store_true',
		help = "Randomise multimodal TMI analysis")

	stat = ap.add_mutually_exclusive_group(required=False)
	stat.add_argument("-glm","--generalizedlinearmodel",
		action='store_true')
	stat.add_argument("-med","--mediation",
		action='store_true')
	stat.add_argument("-ofa","--onebetweenssubjectfactor",
		action='store_true')
	stat.add_argument("-tfa","--twobetweenssubjectfactor",
		action='store_true')
	stat.add_argument("-cos","--cosinor",
		action='store_true')
	stat.add_argument("-mcos","--cosinormediation",
		action='store_true')
	ap.add_argument("-e", "--exchangeblock", 
		nargs=1, 
		help="Exchangability blocks", 
		metavar=('*.csv'), 
		required=False)

	return ap

def run(opts):

	arg_perm_start = int(opts.range[0])
	arg_perm_stop = int(opts.range[1]) + 1
	if opts.surface:
		surface = str(opts.surface[0])

	if opts.generalizedlinearmodel:
		if opts.surface:
			tempdir = "tmtemp_GLM_%s" % surface
			outdir = "output_GLM_%s/perm_GLM" % surface
		else:
			tempdir = "tmtemp_GLM_volume"
			outdir = "output_GLM_volume/perm_GLM" 
		exog_flat = np.load("%s/exog_flat.npy" % tempdir)
		exog_shape = np.load("%s/exog_shape.npy" % tempdir)
		count = 0
		exog = []
		for nc in exog_shape:
			exog.append(exog_flat[:,count:(count+nc)])
			count += nc
		varnames = np.load("%s/varnames.npy" % tempdir)
		gstat = np.load("%s/gstat.npy" % tempdir)
	if opts.mediation:
		if opts.surface:
			tempdir = "tmtemp_mediation_%s" % surface
			outdir = "output_mediation_%s/perm_mediation" % surface
		else:
			tempdir = "tmtemp_mediation_volume"
			outdir = "output_mediation_volume/perm_mediation"
		dmy_leftvar = np.load("%s/dmy_leftvar.npy" % tempdir)
		dmy_rightvar = np.load("%s/dmy_rightvar.npy" % tempdir)
		medtype = np.load("%s/medtype.npy" % tempdir)
	if opts.cosinor:
		if opts.surface:
			tempdir = "tmtemp_cosinor_%s" % surface
			outdir = "output_cosinor_%s/perm_cosinor" % surface
		else:
			tempdir = "tmtemp_cosinor_volume"
			outdir = "output_cosinor_volume/perm_cosinor"
		time_var = np.load("%s/time_var.npy" % tempdir)
		period = np.load("%s/period.npy" % tempdir)

	if opts.cosinormediation:
		if opts.surface:
			tempdir = "tmtemp_medcosinor_%s" % surface
			outdir = "output_medcosinor_%s/perm_cosinor" % surface
		else:
			tempdir = "tmtemp_medcosinor_volume"
			outdir = "output_medcosinor_volume/perm_cosinor"
		time_var = np.load("%s/time_var.npy" % tempdir)
		period = np.load("%s/period.npy" % tempdir)
		dmy_mediator = np.load("%s/dmy_mediator.npy" % tempdir)
		medtype = np.load("%s/medtype.npy" % tempdir)
	if opts.onebetweenssubjectfactor:
		tempdir = "tmtemp_rmANCOVA1BS_%s" % surface
		outdir = "output_rmANCOVA1BS_%s/perm_rmANCOVA1BS" % surface
		dmy_factor1 = np.load("%s/dmy_factor1.npy" % tempdir)
		factors = np.load("%s/factors.npy" % tempdir)
		dmy_subjects = np.load("%s/dmy_subjects.npy" % tempdir)
		dformat = np.load("%s/dformat.npy" % tempdir)[0]
	if opts.twobetweenssubjectfactor:
		tempdir = "tmtemp_rmANCOVA2BS_%s" % surface
		outdir = "output_rmANCOVA2BS_%s/perm_rmANCOVA2BS" % surface
		dmy_factor1 = np.load("%s/dmy_factor1.npy" % tempdir)
		dmy_factor2 = np.load("%s/dmy_factor2.npy" % tempdir)
		factors = np.load("%s/factors.npy" % tempdir)
		dmy_subjects = np.load("%s/dmy_subjects.npy" % tempdir)
		dformat = np.load("%s/dformat.npy" % tempdir)[0]

	# common inputs
	data = np.load("%s/data.npy" % tempdir)
	optstfce = np.load("%s/optstfce.npy" % tempdir)
	dmy_covariates = np.load("%s/dmy_covariates.npy" % tempdir)
	if opts.surface:
		num_vertex_lh = np.load("%s/num_vertex_lh.npy" % tempdir)
		mask_lh = np.load("%s/mask_lh.npy" % tempdir)
		mask_rh = np.load("%s/mask_rh.npy" % tempdir)
		adjac_lh = np.load("%s/adjac_lh.npy" % tempdir)
		adjac_rh = np.load("%s/adjac_rh.npy" % tempdir)
		vdensity_lh = np.load("%s/vdensity_lh.npy" % tempdir)
		vdensity_rh = np.load("%s/vdensity_rh.npy" % tempdir)
		calcTFCE_lh = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac_lh)
		calcTFCE_rh = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac_rh)
	else:
		adjac = np.load("%s/adjac.npy" % tempdir)
		calcTFCE = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac)

	if np.all(dmy_covariates) is None:
		dmy_covariates = None

	if opts.exchangeblock:
		block_list = np.genfromtxt(opts.exchangeblock[0], dtype=np.str)
		is_equal_sizes = check_blocks(block_list)


	#permute T values and write max TFCE values
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	os.chdir(outdir)

	if opts.generalizedlinearmodel:
		print("Calculating null distribution for %s statistics" % gstat)

	for iter_perm in range(arg_perm_start,arg_perm_stop):
		print("Iteration number : %d" % (iter_perm))

		# GLM
		if opts.generalizedlinearmodel:
			Tvalues = Fvalues = None

			if opts.exchangeblock:
				rand_array = rand_blocks(block_list, is_equal_sizes)
			else:
				rand_array = np.random.permutation(list(range(data.shape[0])))

			if gstat == 'f':
				Fvalues = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							verbose = False,
							rand_array = rand_array)[1]
			elif gstat == 't':
				Tvalues = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_fvalues = False,
							output_tvalues = True,
							verbose = False,
							rand_array = rand_array)
			else:
				Fvalues, Tvalues = glm_typeI(data,
							exog,
							dmy_covariates = dmy_covariates,
							output_tvalues =True,
							verbose = False,
							rand_array = rand_array)[1:]

			if Tvalues is not None:
				numcon = np.concatenate(exog,1).shape[1]
				for j in range(numcon):
					tnum=j+1
					if opts.surface:
						write_perm_maxTFCE_vertex('Tstat_con%d' % tnum, 
												Tvalues[tnum],
												num_vertex_lh,
												mask_lh,
												mask_rh,
												calcTFCE_lh,
												calcTFCE_rh,
												vdensity_lh,
												vdensity_rh)
						write_perm_maxTFCE_vertex('Tstat_con%d' % tnum, 
												-Tvalues[tnum],
												num_vertex_lh,
												mask_lh,
												mask_rh,
												calcTFCE_lh,
												calcTFCE_rh,
												vdensity_lh,
												vdensity_rh)
					else:
						write_perm_maxTFCE_voxel('Tstat_con%d' % tnum,
												Tvalues[tnum],
												calcTFCE)
						write_perm_maxTFCE_voxel('Tstat_con%d' % tnum,
												-Tvalues[tnum],
												calcTFCE)

			if Fvalues is not None:
				for j in range(Fvalues.shape[0]):
					if opts.surface:
						write_perm_maxTFCE_vertex('Fstat_%s' % varnames[j],
												Fvalues[j],
												num_vertex_lh,
												mask_lh, mask_rh,
												calcTFCE_lh,
												calcTFCE_rh,
												vdensity_lh,
												vdensity_rh)
					else:
						write_perm_maxTFCE_voxel('Fstat_%s' % varnames[j],
												Fvalues[j],
												calcTFCE)

		# cosinor model
		if opts.cosinor:
			if opts.exchangeblock:
				rand_array = rand_blocks(block_list, is_equal_sizes)
			else:
				rand_array = np.random.permutation(list(range(data.shape[0])))
			# get just the stats for randomization
			_, _, _, _, _, _, _, Fmodel, _, tAMPLITUDE, tACROPHASE, _ = glm_cosinor(endog = data, 
																			time_var = time_var,
																			exog = None,
																			dmy_covariates = dmy_covariates,
																			rand_array = rand_array,
																			period = period,
																			calc_MESOR = False)
			if opts.surface:
				write_perm_maxTFCE_vertex('Fstat_model',
										Fmodel,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
#				write_perm_maxTFCE_vertex('Tstat_mesor',
#										tMESOR,
#										num_vertex_lh,
#										mask_lh,
#										mask_rh,
#										calcTFCE_lh,
#										calcTFCE_rh,
#										vdensity_lh,
#										vdensity_rh)
				for i, per in enumerate(period):
					write_perm_maxTFCE_vertex('Tstat_amplitude_%2.2f' % per,
											tAMPLITUDE[i],
											num_vertex_lh,
											mask_lh,
											mask_rh,
											calcTFCE_lh,
											calcTFCE_rh,
											vdensity_lh,
											vdensity_rh)
					write_perm_maxTFCE_vertex('Tstat_acrophase_%2.2f' % per,
											tACROPHASE[i],
											num_vertex_lh,
											mask_lh,
											mask_rh,
											calcTFCE_lh,
											calcTFCE_rh,
											vdensity_lh,
											vdensity_rh)
			else:
				write_perm_maxTFCE_voxel('Fstat_model',
										Fmodel,
										calcTFCE)
#				write_perm_maxTFCE_voxel('Tstat_mesor',
#										tMESOR,
#										calcTFCE)
				for i, per in enumerate(period):
					write_perm_maxTFCE_voxel('Tstat_amplitude_%2.2f' % per,
											tAMPLITUDE[i],
											calcTFCE)
					write_perm_maxTFCE_voxel('Tstat_acrophase_%2.2f' % per,
											tACROPHASE[i],
											calcTFCE)

		# Cosinor mediation
		if opts.cosinormediation:
			if opts.exchangeblock:
				rand_array = rand_blocks(block_list, is_equal_sizes)
			else:
				rand_array = np.random.permutation(list(range(dmy_leftvar.shape[0])))

			EXOG = []
			EXOG.append(dmy_mediator)

#			_, _, _, _, _, _, _, _, _, tAMPLITUDE_A, _, _ = glm_cosinor(endog = dmy_mediator, 
#																		time_var = time_var,
#																		exog = None,
#																		dmy_covariates = None,
#																		rand_array = rand_array,
#																		period = period)
			_, _, _, _, _, _, _, _, _, tAMPLITUDE_A, _, _ = glm_cosinor(endog = dmy_mediator, 
																		time_var = time_var,
																		exog = None,
																		dmy_covariates = None,
																		rand_array = None,
																		period = period)

			_, _, _, _, _, _, _, _, _, _, _, tEXOG_B = glm_cosinor(endog = data, 
																		time_var = time_var,
																		exog = EXOG,
																		dmy_covariates = None,
																		rand_array = rand_array,
																		period = period)

			SobelZ  = calc_indirect(tAMPLITUDE_A[0], tEXOG_B[0], alg = "aroian")

			if opts.surface:
				write_perm_maxTFCE_vertex("Zstat_%s" % medtype,
										SobelZ,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
			else:
				write_perm_maxTFCE_voxel("Zstat_%s" % medtype,
										SobelZ,
										calcTFCE)

		# MEDIATION
		if opts.mediation:
			if opts.exchangeblock:
				rand_array = rand_blocks(block_list, is_equal_sizes)
			else:
				rand_array = np.random.permutation(list(range(dmy_leftvar.shape[0])))

			if (medtype == 'I'):
				dmy_leftvar = dmy_leftvar[rand_array]

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

			if (medtype == 'M'):
				dmy_leftvar = dmy_leftvar[rand_array]

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

			if (medtype == 'Y'):
				dmy_leftvar = dmy_leftvar[rand_array]
				dmy_rightvar = dmy_rightvar[rand_array]

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

			SobelZ  = calc_indirect(Tvalues_A, Tvalues_B, alg = "aroian")

			if opts.surface:
				write_perm_maxTFCE_vertex("Zstat_%s" % medtype,
										SobelZ,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
			else:
				write_perm_maxTFCE_voxel("Zstat_%s" % medtype,
										SobelZ,
										calcTFCE)

		# One between-subject, one within-subject ANCOVA
		if opts.onebetweenssubjectfactor:

			if opts.exchangeblock:
				rand_array = rand_blocks(block_list, is_equal_sizes)
			else:
				rand_array = np.random.permutation(list(range(dmy_factor1.shape[0])))

			F_a, F_s, F_sa = reg_rm_ancova_one_bs_factor(data, 
									dmy_factor1,
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									data_format = dformat,
									output_sig = False,
									verbose = False,
									rand_array = rand_array)
			if opts.surface:
				write_perm_maxTFCE_vertex("Fstat_%s" % factors[0],
										F_a,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
				write_perm_maxTFCE_vertex("Fstat_time",
										F_s,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
				write_perm_maxTFCE_vertex("Fstat_%s.X.time" % factors[0],
										F_sa,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
			else:
				write_perm_maxTFCE_voxel("Fstat_%s" % factors[0],
										F_a,
										calcTFCE)
				write_perm_maxTFCE_voxel("Fstat_time",
										F_s,
										calcTFCE)
				write_perm_maxTFCE_voxel("Fstat_%s.X.time" % factors[0],
										F_sa,
										calcTFCE)

		# Two between-subject, one within-subject ANCOVA
		if opts.twobetweenssubjectfactor:

			if opts.exchangeblock:
				rand_array = rand_blocks(block_list, is_equal_sizes)
			else:
				rand_array = np.random.permutation(list(range(dmy_factor1.shape[0])))

			F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab = reg_rm_ancova_two_bs_factor(data, 
									dmy_factor1,
									dmy_factor2, 
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									data_format = dformat,
									output_sig = False,
									verbose = False,
									rand_array = rand_array)
			if opts.surface:
				write_perm_maxTFCE_vertex("Fstat_%s" % factors[0],
										F_a,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
				write_perm_maxTFCE_vertex("Fstat_%s" % factors[2],
										F_b,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
				write_perm_maxTFCE_vertex("Fstat_%s.X.%s" % (factors[0],factors[2]),
										F_ab,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
				write_perm_maxTFCE_vertex("Fstat_time",
										F_s,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
				write_perm_maxTFCE_vertex("Fstat_%s.X.time" % factors[0],
										F_sa,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
				write_perm_maxTFCE_vertex("Fstat_%s.X.time" % factors[2],
										F_sb,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
				write_perm_maxTFCE_vertex("Fstat_%s.X.%s.X.time" % (factors[0],factors[2]),
										F_sab,
										num_vertex_lh,
										mask_lh,
										mask_rh,
										calcTFCE_lh,
										calcTFCE_rh,
										vdensity_lh,
										vdensity_rh)
			else:
				write_perm_maxTFCE_voxel("Fstat_%s" % factors[0],
										F_a,
										calcTFCE)
				write_perm_maxTFCE_voxel("Fstat_%s" % factors[2],
										F_b,
										calcTFCE)
				write_perm_maxTFCE_voxel("Fstat_%s.X.%s" % (factors[0],factors[2]),
										F_ab,
										calcTFCE)
				write_perm_maxTFCE_voxel("Fstat_time",
										F_s,
										calcTFCE)
				write_perm_maxTFCE_voxel("Fstat_%s.X.time" % factors[0],
										F_sa,
										calcTFCE)
				write_perm_maxTFCE_voxel("Fstat_%s.X.time" % factors[2],
										F_sb,
										calcTFCE)
				write_perm_maxTFCE_voxel("Fstat_%s.X.%s.X.time" % (factors[0],factors[2]),
										F_sab,
										calcTFCE)
	print(("Finished. Randomization took %.1f seconds" % (time() - start_time)))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
