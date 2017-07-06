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

def get_script_path():
	return os.path.dirname(os.path.realpath(sys.argv[0]))

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument("-i", "--input", 
		nargs=2, 
		help="[Predictor(s)] [Covariate(s)] (recommended)", 
		metavar=('*.csv', '*.csv'))
	group.add_argument("-r", "--regressors", 
		nargs=1, help="Single step regression", 
		metavar=('*.csv'))
	ap.add_argument("-i_tmi", "--tmifile",
		help="Vertex analysis. Input surface: e.g. --vertex [area or thickness]", 
		nargs=1,
		metavar=('stats*.tmi'),
		required=True)
	ap.add_argument("-sa", "--setadjacencyobjs",
		help="Specify the adjaceny object to use for each mask. The number of inputs must match the number of masks in the tmi file. Note, the objects start at zero. e.g., -sa 0 1 0 1",
		nargs='+',
		type=str,
		metavar=('int'))
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
	parallel.add_argument("-c","--condor", 
		help="Use HTCondor.", 
		action="store_true")
	parallel.add_argument("-f","--fslsub", 
		help="Use fsl_sub script.",
		action="store_true")
	return ap

def run(opts):

	currentTime=int(time())

	#load the proper script

	whichScript="python %s/tm_multimodality_multisurface_regression.py" % get_script_path()

	#round number of permutations to the nearest 200
	roundperm=int(np.round(opts.numperm[0]/200.0)*100.0)
	forperm=(roundperm/100)-1
	print "Evaluating %d permuations" % (roundperm*2)

	if opts.input:
		for i in xrange(forperm+1):
			if opts.setadjacencyobjs:
				os.system("echo %s -i %s %s -i_tmi %s -p %i %i -sa %s >> cmd_MStmi_randomise_%d" % (whichScript,opts.input[0], opts.input[1], opts.tmifile[0], (i*100+1), (i*100+100), ' '.join(opts.setadjacencyobjs), currentTime))
			else:
				os.system("echo %s -i %s %s -i_tmi %s -p %i %i >> cmd_MStmi_randomise_%d" % (whichScript,opts.input[0], opts.input[1], opts.tmifile[0], (i*100+1), (i*100+100), currentTime))
	else:
		for i in xrange(forperm+1):
			if opts.setadjacencyobjs:
				os.system("echo %s -r %s -i_tmi %s -p %i %i -sa %s >> cmd_MStmi_randomise_%d" % (whichScript, opts.regressors[0], opts.tmifile[0], (i*100+1), (i*100+100), ' '.join(opts.setadjacencyobjs), currentTime))
			else:
				os.system("echo %s -r %s -i_tmi %s -p %i %i >> cmd_MStmi_randomise_%d" % (whichScript, opts.regressors[0], opts.tmifile[0], (i*100+1), (i*100+100),currentTime) )
	#build command text file

	#submit text file for parallel processing; submit_condor_jobs_file is supplied with TFCE_mediation
	if opts.gnuparallel:
		os.system("cat cmd_MStmi_randomise_%d | parallel -j %d --delay 20" % (currentTime,int(opts.gnuparallel[0])) )
	elif opts.condor:
		os.system("submit_condor_jobs_file cmd_MStmi_randomise_%d" % (currentTime) )
	elif opts.fslsub:
		os.system("${FSLDIR}/bin/fsl_sub -t cmd_MStmi_randomise_%d" % (currentTime) )
	else:
		print "Submit cmd_MStmi_randomise_%d to your job clustering platform for randomisation." % (currentTime)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
