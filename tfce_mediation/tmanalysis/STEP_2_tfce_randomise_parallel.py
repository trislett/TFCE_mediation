#!/usr/bin/env python

#    Wrapper for parallelizing randomise multiple regression with TFCE
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
	datatype = ap.add_mutually_exclusive_group(required=True)
	datatype.add_argument("--voxel", 
		help="Voxel analysis.", 
		action="store_true")
	datatype.add_argument("--vertex", 
		help="Vertex analysis. Input surface: e.g. --vertex {surface}", 
		nargs=1,
		metavar=('surface'))

	stat = ap.add_mutually_exclusive_group(required=False)
	stat.add_argument("-m", "--mediation",
		help="Mediation analysis [M or I or Y]. If not specified, then multiple regression is performed.",
		nargs=1, 
		choices = ['I','M','Y'], 
		metavar=('STR'))
	stat.add_argument("-glm","--generalizedlinearmodel",
		help="GLM analysis with tm-models.",
		action='store_true')
	stat.add_argument("-med","--modelmediation",
		help="Mediation with tm-models",
		action='store_true')
	stat.add_argument("-ofa","--onebetweenssubjectfactor",
		help="One factor repeated measure ANCOVA with tm-models",
		action='store_true')
	stat.add_argument("-tfa","--twobetweenssubjectfactor",
		help="Two factor repeated measure ANCOVA with tm-models",
		action='store_true')
	stat.add_argument("-cos","--cosinor",
		help="Cosinor model",
		action='store_true')
	stat.add_argument("-mcos","--cosinormediation",
		action='store_true')

	ap.add_argument("-n", "--numperm", 
		nargs=1, 
		type=int, 
		help="# of permutations", 
		metavar=('INT'), 
		required=True)
	ap.add_argument("-v", "--specifyvars", 
		nargs=2, type=int, 
		help="Optional for multiple regression. Specify which regressors are permuted [first] [last]. For one variable, first=last.", 
		metavar=('INT','INT'))
	ap.add_argument("-e", "--exchangeblock", 
		nargs=1, 
		help="Exchangability blocks", 
		metavar=('*.csv'), 
		required=False)
	group = ap.add_mutually_exclusive_group(required=False)
	group.add_argument("-p","--gnuparallel", 
		nargs=1, 
		type=int, 
		help="Use GNU parallel. Specify number of cores", 
		metavar=('INT'))
	group.add_argument("-c","--condor", 
		help="Use HTCondor.", 
		action="store_true")
	group.add_argument("-f","--fslsub", 
		help="Use fsl_sub script.",
		action="store_true")
	return ap

def run(opts):

	currentTime=int(time())

	#load the proper script
	if opts.voxel:
		whichScript="tfce_mediation voxel-regress-randomise"
		if opts.specifyvars:
			whichScript="tfce_mediation voxel-regress-randomise -v %d %d" % (opts.specifyvars[0], opts.specifyvars[1])
		if opts.mediation:
			whichScript= "tfce_mediation voxel-mediation-randomise -m %s" % (opts.mediation[0])
		if opts.generalizedlinearmodel:
			whichScript= "tfce_mediation tm-models-randomise -v -glm"
		if opts.onebetweenssubjectfactor:
			whichScript= "tfce_mediation tm-models-randomise -v -ofa"
		if opts.twobetweenssubjectfactor:
			whichScript= "tfce_mediation tm-models-randomise -v -tfa"
		if opts.cosinor:
			whichScript= "tfce_mediation tm-models-randomise -v -cos"
		if opts.modelmediation:
			whichScript= "tfce_mediation tm-models-randomise -v -med"
		if opts.cosinormediation:
			whichScript= "tfce_mediation tm-models-randomise -v -mcos"
	else:
		whichScript= "tfce_mediation vertex-regress-randomise -s %s" % (opts.vertex[0])
		if opts.specifyvars:
			whichScript= "tfce_mediation vertex-regress-randomise -s %s -v %d %d" % (opts.vertex[0], opts.specifyvars[0], opts.specifyvars[1])
		if opts.mediation:
			whichScript = "tfce_mediation vertex-mediation-randomise -s %s -m %s" % (opts.vertex[0],opts.mediation[0])
		if opts.generalizedlinearmodel:
			whichScript = "tfce_mediation tm-models-randomise -s %s -glm" % (opts.vertex[0])
		if opts.onebetweenssubjectfactor:
			whichScript = "tfce_mediation tm-models-randomise -s %s -ofa" % (opts.vertex[0])
		if opts.twobetweenssubjectfactor:
			whichScript = "tfce_mediation tm-models-randomise -s %s -tfa" % (opts.vertex[0])
		if opts.cosinor:
			whichScript= "tfce_mediation tm-models-randomise -s %s -cos" % (opts.vertex[0])
		if opts.modelmediation:
			whichScript = "tfce_mediation tm-models-randomise -s %s -med" % (opts.vertex[0])
		if opts.cosinormediation:
			whichScript= "tfce_mediation tm-models-randomise -s %s  -mcos" % (opts.vertex[0])

	if opts.exchangeblock:
		whichScript = "%s -e %s" % (whichScript, opts.exchangeblock[0])


	#round number of permutations to the nearest 200
	roundperm=int(np.round(opts.numperm[0]/200.0)*100.0)
	print("Evaluating %d permuations" % (roundperm*2))
	if opts.mediation or opts.onebetweenssubjectfactor or opts.twobetweenssubjectfactor or opts.cosinor or opts.cosinormediation:
		roundperm *= 2
	forperm=int(roundperm/100)-1

	#build command text file
	for i in range(forperm+1):
		os.system("echo %s -r %i %i >> cmd_TFCE_randomise_%d" % (whichScript, (i*100+1), (i*100+100),currentTime) )


	#submit text file for parallel processing; submit_condor_jobs_file is supplied with TFCE_mediation
	if opts.gnuparallel:
		os.system("cat cmd_TFCE_randomise_%d | parallel -j %d" % (currentTime,int(opts.gnuparallel[0])) )
	elif opts.condor:
		os.system("submit_condor_jobs_file cmd_TFCE_randomise_%d" % (currentTime) )
	elif opts.fslsub:
		os.system("${FSLDIR}/bin/fsl_sub -t cmd_TFCE_randomise_%d" % (currentTime) )

	if opts.voxel:
		print("Run: tfce_mediation voxel-calculate-fwep to calculate (1-P[FWE]) image (after randomisation is finished).")
	else:
		print("Run: tfce_mediation vertex-calculate-fwep to calculate (1-P[FWE]) image (after randomisation is finished).")

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
