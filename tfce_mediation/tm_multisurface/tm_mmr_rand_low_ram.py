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

from __future__ import division
import os
import numpy as np
import argparse as ap
from time import time

from tfce_mediation.cynumstats import resid_covars
from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.tm_io import read_tm_filetype, write_tm_filetype, savemgh_v2, savenifti_v2
from tfce_mediation.pyfunc import save_ply, convert_voxel, vectorized_surface_smooth
from tfce_mediation.tm_func import calculate_tfce, calculate_mediation_tfce, calc_mixed_tfce, apply_mfwer, create_full_mask, merge_adjacency_array, lowest_length, create_position_array, paint_surface, strip_basename, saveauto

DESCRIPTION = "Description"

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
	ap.add_argument("-n", "--numperm", 
		nargs=1, 
		type=int, 
		help="# of permutations", 
		metavar=('INT'), 
		required=True)
	parallel = ap.add_mutually_exclusive_group(required=False)
	parallel.add_argument("-p","--gnuparallel", 
		nargs=1, 
		type=int, 
		help="Use GNU parallel. Specify number of cores", 
		metavar=('INT'))
	parallel.add_argument("-cd","--condor", 
		help="Use HTCondor.", 
		action="store_true")
	parallel.add_argument("-f","--fslsub", 
		help="Use fsl_sub script.",
		action="store_true")
	parallel.add_argument("-t","--cmdtext", 
		help="Outputs a text file with one command per line.",
		action="store_true")
	return ap


def run(opts):
	currentTime=int(time())
	temp_directory = "tmi_temp"
	if not os.path.exists(temp_directory):
		os.mkdir(temp_directory)
	_, image_array, masking_array, _, _, _, _, _, adjacency_array, _, _  = read_tm_filetype(opts.tmifile[0])
	position_array = create_position_array(masking_array)

	if opts.setadjacencyobjs:
		if len(opts.setadjacencyobjs) == len(masking_array):
			adjacent_range = np.array(opts.setadjacencyobjs, dtype = np.int)
		else:
			print "Error: # of masking arrays (%d) must and list of matching adjacency (%d) must be equal." % (len(masking_array), len(opts.setadjacencyobjs))
			quit()
	else: 
		adjacent_range = range(len(adjacency_array))

	for i in range(len(masking_array)):
		if not opts.noweight:
			temp_vdensity = np.zeros((adjacency_array[adjacent_range[i]].shape[0]))
			for j in xrange(adjacency_array[adjacent_range[i]].shape[0]):
				temp_vdensity[j] = len(adjacency_array[adjacent_range[i]][j])
			if masking_array[i].shape[2] == 1:
				temp_vdensity = temp_vdensity[masking_array[i][:,0,0]==True]
		else:
			temp_vdensity = np.array([1])
		np.save("%s/%s_vdensity_temp.npy" % (temp_directory, i), temp_vdensity)
		temp_vdensity = None
	for num, j in enumerate(adjacent_range):
		np.save("%s/%s_adjacency_temp.npy" % (temp_directory, num), np.copy(adjacency_array[j]))
	if opts.covariates:
		covars = np.genfromtxt(opts.covariates[0], delimiter=',')
		x_covars = np.column_stack([np.ones(len(covars)),covars])
	for data_count in range(len(masking_array)):
		start = position_array[data_count]
		end = position_array[data_count+1]
		data_array = image_array[0][start:end,:]

		if opts.subset:
			masking_variable = np.isfinite(np.genfromtxt(str(opts.subset[0]), delimiter=','))
			if opts.covariates:
				merge_y = resid_covars(x_covars,data_array[:,masking_variable])
			else:
				merge_y = data_array[:,masking_variable].T 
		else:
			if opts.covariates:
				merge_y = resid_covars(x_covars,data_array)
			else:
				merge_y = data_array.T
		np.save("%s/%s_data_temp.npy" % (temp_directory, data_count), merge_y)
		merge_y = data_array = None
	np.save("%s/options.npy" % temp_directory,opts)


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
