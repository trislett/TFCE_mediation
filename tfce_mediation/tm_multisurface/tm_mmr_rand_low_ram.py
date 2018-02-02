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
from tfce_mediation.tm_func import calculate_tfce, calculate_mediation_tfce, calc_mixed_tfce, apply_mfwer, create_full_mask, merge_adjacency_array, lowest_length, create_position_array, paint_surface, strip_basename, saveauto, low_ram_calculate_tfce

DESCRIPTION = "mmr-lr: lower ram requirement version of tm_multimodal mmr (multimodality, multisurface regression)."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):

	ap.add_argument("-i_tmi", "--tmifile",
		help="Input the *.tmi file for analysis.", 
		nargs=1,
		metavar=('*.tmi'),
		required=True)
	ap.add_argument("-os", "--outputstats",  
		help = "Calculates the stats without permutation testing, and outputs the tmi", 
		action =('store_true'))
	ap.add_argument("-n", "--numperm", 
		nargs=1,
		type=int,
		help="# of permutations",
		metavar=('INT'))
	ap.add_argument("-um", "--usepreviousmemorymapping", 
		help="Skip the creation of memory mapped objects. It will give an error if tmi_temp folder does not exist.",
		action="store_true")

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
	parallel.add_argument("--serial", 
		help="Runs the command sequentially using one core (not recommended)",
		action="store_true")
	return ap


def run(opts):
	currentTime=int(time())
	temp_directory = "tmi_temp"
	if not os.path.exists(temp_directory):
		if opts.usepreviousmemorymapping:
			print("Error: tmi_temp folder not found. Change directory, or do not use -um argument")
			quit()
		else:
			os.mkdir(temp_directory)

	# permutation options
	if opts.outputstats:
		_, image_array, masking_array, maskname, affine_array, vertex_array, face_array, surfname, adjacency_array, _, _  = read_tm_filetype(opts.tmifile[0])
	else:
		if opts.serial:
			parallel = 'serial'
		elif opts.cmdtext:
			parallel = 'cmdtext'
		elif opts.fslsub:
			parallel = 'fslsub'
		elif opts.condor:
			parallel = 'condor'
		elif opts.gnuparallel:
			parallel = 'gnuparallel'
		else:
			print("Either {-os} options must be used, or {-n} with a parallelization option { -p # | -cd | -d | -t | --serial } must be used with permutation testing.")
			quit()

		_, image_array, masking_array, _, _, _, _, _, adjacency_array, _, _  = read_tm_filetype(opts.tmifile[0])
	position_array = create_position_array(masking_array)

	if opts.setadjacencyobjs:
		if len(opts.setadjacencyobjs) == len(masking_array):
			adjacent_range = np.array(opts.setadjacencyobjs, dtype = np.int)
		else:
			print("Error: # of masking arrays (%d) must and list of matching adjacency (%d) must be equal." % (len(masking_array), len(opts.setadjacencyobjs)))
			quit()
	else: 
		adjacent_range = list(range(len(adjacency_array)))

	if opts.usepreviousmemorymapping:
		if os.path.isfile("%s/opts.npy" % temp_directory):
			print("Renaming previous %s/opts.npy to %s/backup_%d_opts.npy" % (temp_directory,temp_directory, currentTime))
			os.system("mv %s/opts.npy %s/backup_%d_opts.npy" % (temp_directory,temp_directory, currentTime))
		np.save("%s/opts.npy" % temp_directory, opts)
	else:
		for i in range(len(masking_array)):
			if not opts.noweight:
				temp_vdensity = np.zeros((adjacency_array[adjacent_range[i]].shape[0]))
				for j in range(adjacency_array[adjacent_range[i]].shape[0]):
					temp_vdensity[j] = len(adjacency_array[adjacent_range[i]][j])
				if masking_array[i].shape[2] == 1:
					temp_vdensity = temp_vdensity[masking_array[i][:,0,0]==True]
			else:
				temp_vdensity = np.array([1])
			vdensity = np.array((1 - (temp_vdensity/temp_vdensity.max())+(temp_vdensity.mean()/temp_vdensity.max())), dtype=np.float32)
			np.save("%s/%s_vdensity_temp.npy" % (temp_directory, i), vdensity)


			if masking_array[i].shape[2] == 1: # check if vertex or voxel image
				outmask = masking_array[i][:,0,0]
			else:
				outmask = masking_array[i][masking_array[i]==True]


			np.save("%s/%s_mask_temp.npy" % (temp_directory, i), outmask)
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
			np.save("%s/%s_data_temp.npy" % (temp_directory, data_count), merge_y.astype(np.float32, order = "C"))
			merge_y = data_array = None
		np.save("%s/opts.npy" % temp_directory, opts)
		np.save("%s/masking_array.npy" % temp_directory, masking_array)

	outname = opts.tmifile[0][:-4]
	# make output folder
	if not os.path.exists("output_%s" % (outname)):
		os.mkdir("output_%s" % (outname))

	#run the stats
	if opts.outputstats:
		np.save("%s/masking_array.npy" % temp_directory, masking_array)
		np.save("%s/maskname.npy" % temp_directory, maskname)
		np.save("%s/affine_array.npy" % temp_directory, affine_array)
		np.save("%s/vertex_array.npy" % temp_directory, vertex_array)
		np.save("%s/face_array.npy" % temp_directory, face_array)
		np.save("%s/surfname.npy" % temp_directory, surfname)

		if opts.inputmediation:
			output_dir = "output_%s/output_%s_med_stats_%s.tmi" % (outname, str(opts.inputmediation[0]), outname)
			if not os.path.exists(output_dir):
				os.mkdir(output_dir)
			print("OUTPUT DIRECTORY: %s" % (output_dir))
		else:
			output_dir = "output_%s/output_stats_%s.tmi" % (outname, outname)
			if not os.path.exists(output_dir):
				os.mkdir(output_dir)
			print("OUTPUT DIRECTORY: %s" % (output_dir))
		print("Writing statistics TMI file")
		os.system("tm_multimodal mmr-lr-run --path %s -os" % (output_dir))
	else:
		output_dir = "output_%s/output_stats_%s.tmi" % (outname, outname)
		if not os.path.isfile("output_%s/stats_%s.tmi" % (outname, outname)):
			print("Warning: output_%s/stats_%s.tmi file not detected.\nTo create the output tmi: (1) use the --noperm argument or (2) run tm_multimodal mmr (recommended)" % (outname, outname))
			if not os.path.exists(output_dir):
				os.mkdir(output_dir)
		print("OUTPUT DIRECTORY: o%s" % (output_dir))

	if opts.numperm: 
		mmr_cmd = "echo tm_multimodal mmr-lr-run --path %s " % (output_dir)
		#round number of permutations to the nearest 200
		roundperm = int(np.round(opts.numperm[0]/200.0) * 100.0)
		forperm = int((roundperm/100) - 1)
		print("Evaluating %d permuations" % (roundperm*2))

		#build command text file
		for i in range(forperm+1):
			random_seed = int(i+int(float(str(time())[-6:])*100))
			print("Block %d Seed:\t%d" % (i, random_seed))
			for j in range(len(masking_array)):
				os.system("%s -sn %d -pr %i %i --seed %i >> cmd_MStmi_randomise_%d" % (mmr_cmd, j, (i*100+1), (i*100+100), random_seed, currentTime))

		print("Submitting jobs for parallel processing")
		#submit text file for parallel processing; submit_condor_jobs_file is supplied with TFCE_mediation
		if opts.gnuparallel:
			os.system("cat cmd_MStmi_randomise_%d | parallel -j %d" % (currentTime, int(opts.gnuparallel[0])))
		elif opts.condor:
			os.system("submit_condor_jobs_file cmd_MStmi_randomise_%d" % (currentTime))
		elif opts.fslsub:
			os.system("${FSLDIR}/bin/fsl_sub -t cmd_MStmi_randomise_%d" % (currentTime))
		elif opts.serial:
			os.system("cat cmd_MStmi_randomise_%d | parallel -j 1" % (currentTime))
		elif opts.cmdtext:
			pass
		else:
			print("Submit cmd_MStmi_randomise_%d to your job clustering platform for randomisation." % (currentTime))


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
