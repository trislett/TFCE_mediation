#!/usr/bin/env python

#    Wrapper for parallelizing multimodality_multisurfce_regression.
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
import numpy as np
import argparse as ap
from time import time

DESCRIPTION = "Different parallelization methods for TFCE_mediation permutation testing. If no parallelization method is specified, only a text file of commands will be outputed (i.e., cmd_TFCE_randomise_{timestamp})"
formatter_class=lambda prog: ap.HelpFormatter(prog, max_help_position=100, width=200)

def get_script_path():
	return os.path.dirname(os.path.realpath(sys.argv[0]))

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=formatter_class)):
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
	ap.add_argument("-i_tmi", "--tmifile",
		help="Vertex analysis. Input surface: e.g. --vertex [area or thickness]", 
		nargs=1,
		metavar=('*.tmi'),
		required=True)
	ap.add_argument("--tfce", 
		help="TFCE settings. H (i.e., height raised to power H), E (i.e., extent raised to power E). Default: %(default)s). H=2, E=2/3. Multiple sets of H and E values can be entered with using the -st option.", 
		nargs='+', 
		type=str,
		metavar=('H', 'E'))
	ap.add_argument("-sa", "--setadjacencyobjs",
		help="Specify the adjaceny object to use for each mask. The number of inputs must match the number of masks in the tmi file. Note, the objects start at zero. e.g., -sa 0 1 0 1",
		nargs='+',
		type=str,
		metavar=('int'))
	ap.add_argument("-st", "--assigntfcesettings",
		help="Specify the tfce H and E settings for each mask. -st is useful for combined analysis do voxel and vertex data. More than one set of values must inputted with --tfce. The number of inputs must match the number of masks in the tmi file. The input corresponds to each pair of --tfce setting starting at zero. e.g., -st 0 0 0 0 1 1",
		nargs='+',
		type=str,
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

	#assign command options
	mmr_cmd = "echo tm_multimodal mmr -i_tmi %s" % opts.tmifile[0]
	if opts.input:
		cmd_input = " -i"
		for infiles in opts.input:
			cmd_input += " %s" % infiles
		mmr_cmd += cmd_input
	elif opts.inputmediation:
		mmr_cmd += " -im %s %s %s" % (opts.inputmediation[0], opts.inputmediation[1], opts.inputmediation[2])
	else:
		mmr_cmd += " -r %s" % (opts.regressors[0])
	if opts.covariates:
		mmr_cmd += " -c %s" % (opts.covariates[0])
	if opts.tfce:
		mmr_cmd += " --tfce %s" % ' '.join(opts.tfce)
	if opts.setadjacencyobjs:
		mmr_cmd += " -sa %s" % ' '.join(opts.setadjacencyobjs)
	if opts.assigntfcesettings:
		if not opts.tfce:
			print("Error: --tfce must be used with -st option.")
			quit()
		mmr_cmd += " -st %s" % ' '.join(opts.assigntfcesettings)
	if opts.noweight:
		mmr_cmd += " --noweight"
	if opts.subset:
		mmr_cmd += " --subset %s" % (opts.subset[0])


	#round number of permutations to the nearest 200
	roundperm=int(np.round(opts.numperm[0]/200.0)*100.0)
	forperm=(roundperm/100)-1
	print("Evaluating %d permuations" % (roundperm*2))

	#build command text file
	for i in range(forperm+1):
		os.system("%s -p %i %i >> cmd_MStmi_randomise_%d" % (mmr_cmd, (i*100+1), (i*100+100), currentTime))

	#submit text file for parallel processing; submit_condor_jobs_file is supplied with TFCE_mediation
	if opts.gnuparallel:
		os.system("cat cmd_MStmi_randomise_%d | parallel -j %d --delay 20" % (currentTime, int(opts.gnuparallel[0])))
	elif opts.condor:
		os.system("submit_condor_jobs_file cmd_MStmi_randomise_%d" % (currentTime))
	elif opts.fslsub:
		os.system("${FSLDIR}/bin/fsl_sub -t cmd_MStmi_randomise_%d" % (currentTime))
	elif opts.cmdtext:
		pass
	else:
		print("Submit cmd_MStmi_randomise_%d to your job clustering platform for randomisation." % (currentTime))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
