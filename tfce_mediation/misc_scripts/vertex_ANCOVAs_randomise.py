#!/usr/bin/env python

#    Multiple regression with TFCE
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
import numpy as np
import argparse as ap

from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.pyfunc import reg_rm_ancova_one_bs_factor, reg_rm_ancova_two_bs_factor, glm_typeI, write_perm_maxTFCE_vertex

DESCRIPTION = "Vertex-wise GLMs include within-subject interactions with TFCE."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-r", "--range", 
		nargs=2, 
		type=int, 
		help="permutation [start] [stop]", 
		metavar=('INT','INT'), 
		required=True)
	ap.add_argument("-s", "--surface", 
		nargs=1, 
		help="surface (area or thickness)", 
		metavar=('STR'), 
		required=True)

	stat = ap.add_mutually_exclusive_group(required=False)
	stat.add_argument("-glm","--generalizedlinearmodel",
		action='store_true')
	stat.add_argument("-of","--onebetweenssubjectfactor",
		action='store_true')
	stat.add_argument("-tf","--twobetweenssubjectfactor",
		action='store_true')

	return ap

def run(opts):

	arg_perm_start = int(opts.range[0])
	arg_perm_stop = int(opts.range[1]) + 1
	surface = str(opts.surface[0])

	if opts.generalizedlinearmodel:
		tempdir = "tmp_tmGLM_%s" % surface
		outdir = "outputGLM_%s/perm_GLM" % surface
		exog_flat = np.load("%s/exog_flat.npy" % tempdir)
		exog_shape = np.load("%s/exog_shape.npy" % tempdir)
		count = 0
		exog = []
		for nc in exog_shape:
			exog.append(exog_flat[:,count:(count+nc)])
			count += nc
		varnames = np.load("%s/varnames.npy" % tempdir)
	if opts.onebetweenssubjectfactor:
		tempdir = "tmp_tmANCOVA1BS_%s" % surface
		outdir = "outputANCOVA1BS_%s/perm_ANCOVA1BS" % surface
		dmy_factor1 = np.load("%s/dmy_factor1.npy" % tempdir)
		factors = np.load("%s/factors.npy" % tempdir)
		dmy_subjects = np.load("%s/dmy_subjects.npy" % tempdir)
	if opts.twobetweenssubjectfactor:
		tempdir = "tmp_tmANCOVA2BS_%s" % surface
		outdir = "outputANCOVA2BS_%s/perm_ANCOVA2BS" % surface
		dmy_factor1 = np.load("%s/dmy_factor1.npy" % tempdir)
		dmy_factor2 = np.load("%s/dmy_factor2.npy" % tempdir)
		factors = np.load("%s/factors.npy" % tempdir)
		dmy_subjects = np.load("%s/dmy_subjects.npy" % tempdir)
	# common inputs
	num_vertex_lh = np.load("%s/num_vertex_lh.npy" % tempdir)
	mask_lh = np.load("%s/mask_lh.npy" % tempdir)
	mask_rh = np.load("%s/mask_rh.npy" % tempdir)
	adjac_lh = np.load("%s/adjac_lh.npy" % tempdir)
	adjac_rh = np.load("%s/adjac_rh.npy" % tempdir)
	data = np.load("%s/data.npy" % tempdir)
	dmy_covariates = np.load("%s/dmy_covariates.npy" % tempdir)
	optstfce = np.load("%s/optstfce.npy" % tempdir)
	vdensity_lh = np.load("%s/vdensity_lh.npy" % tempdir)
	vdensity_rh = np.load("%s/vdensity_rh.npy" % tempdir)

	if np.all(dmy_covariates) is None:
		dmy_covariates = None

	#load TFCE fucntion
	calcTFCE_lh = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac_lh) # H=2, E=1
	calcTFCE_rh = CreateAdjSet(float(optstfce[0]), float(optstfce[1]), adjac_rh) # H=2, E=1

	#permute T values and write max TFCE values
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	os.chdir(outdir) 

	for iter_perm in range(arg_perm_start,arg_perm_stop):
		print("Iteration number : %d" % (iter_perm))

		if opts.generalizedlinearmodel:
			rand_array = np.random.permutation(list(range(data.shape[0])))
			Fmodel, Fvalues = glm_typeI(data,
						exog,
						dmy_covariates = dmy_covariates,
						verbose = False,
						rand_array = rand_array)

			write_perm_maxTFCE_vertex('F_model', Fmodel, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)

			for j in range(Fvalues.shape[0]):
				write_perm_maxTFCE_vertex('F_%s' % varnames[j], Fvalues[j], num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)

		if opts.onebetweenssubjectfactor:
			rand_array = np.random.permutation(list(range(data.shape[1])))
			F_a, F_s, F_sa = reg_rm_ancova_one_bs_factor(data, 
									dmy_factor1,
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False,
									verbose = False,
									rand_array = rand_array)
			write_perm_maxTFCE_vertex("Fstat_%s" % factors[0], F_a, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)
			write_perm_maxTFCE_vertex("Fstat_time", F_s, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)
			write_perm_maxTFCE_vertex("Fstat_%s.X.time" % factors[0], F_sa, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)

		if opts.twobetweenssubjectfactor:
			rand_array = np.random.permutation(list(range(data.shape[1])))
			F_a, F_b, F_ab, F_s, F_sa, F_sb, F_sab = reg_rm_ancova_two_bs_factor(data, 
									dmy_factor1,
									dmy_factor2, 
									dmy_subjects,
									dmy_covariates = dmy_covariates,
									output_sig = False,
									verbose = False,
									rand_array = rand_array)
			write_perm_maxTFCE_vertex("Fstat_%s" % factors[0], F_a, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)
			write_perm_maxTFCE_vertex("Fstat_%s" % factors[2], F_b, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)
			write_perm_maxTFCE_vertex("Fstat_%s.X.%s" % (factors[0],factors[2]), F_ab, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)
			write_perm_maxTFCE_vertex("Fstat_time", F_s, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)
			write_perm_maxTFCE_vertex("Fstat_%s.X.time" % factors[0], F_sa, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)
			write_perm_maxTFCE_vertex("Fstat_%s.X.time" % factors[2], F_sb, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)
			write_perm_maxTFCE_vertex("Fstat_%s.X.%s.X.time" % (factors[0],factors[2]), F_sb, num_vertex_lh, mask_lh, mask_rh, calcTFCE_lh, calcTFCE_rh, vdensity_lh, vdensity_rh)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
