#!/usr/bin/python

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

def get_script_path():
	return os.path.dirname(os.path.realpath(sys.argv[0]))
def round_perm_num(numperm):
	return ((numperm + 50)/100)*100/2

ap = ap.ArgumentParser(description="Permutation testing for multiple regression with TFCE on vertex data.")
ap.add_argument("-n", "--numperm", nargs=1, type=int, help="# of permutations", metavar=('INT'), required=True)
ap.add_argument("-s", "--surface", nargs=1, help="surface (area or thickness)", metavar=('STR'), required=True)
ap.add_argument("-v", "--specifyvars", nargs=2, type=int, help="Optional. Specify which regressors are permuted [first] [last]. For one variable, first=last.", metavar=('INT','INT'))
group = ap.add_mutually_exclusive_group(required=True)
group.add_argument("-p","--gnuparallel", nargs=1, type=int, help="Use GNU parallel. Specify number of cores", metavar=('INT'))
group.add_argument("-c","--condor", help="Use HTCondor.", action="store_true")
group.add_argument("-f","--fslsub", help="Use fsl_sub script.",action="store_true")
opts = ap.parse_args()


SCRIPTPATH=get_script_path()
currentTime=int(time())

roundperm=int(np.round(opts.numperm[0]/200.0)*100.0)
forperm=(roundperm/100)-1
print "Evaluating %d permuations" % (roundperm*2)

if opts.specifyvars:
	for i in xrange(forperm+1):
		os.system("echo %s/vertex_tfce_multiple_regression_randomise.py -r %i %i -s %s -v %d %d >> cmd_multipleregress_randomise_%s_%d" % (SCRIPTPATH, (i*100+1), (i*100+100), opts.surface[0], int(opts.specifyvars[0]), int(opts.specifyvars[1]) ,opts.surface[0],currentTime) ) 
else:
	for i in xrange(forperm+1):
		os.system("echo %s/vertex_tfce_multiple_regression_randomise.py -r %i %i -s %s >> cmd_multipleregress_randomise_%s_%d" % (SCRIPTPATH, (i*100+1), (i*100+100), opts.surface[0],opts.surface[0],currentTime) ) 


if opts.gnuparallel:
	os.system("cat cmd_multipleregress_randomise_%s_%d | parallel -j %d" % (opts.surface[0],currentTime,int(opts.gnuparallel[0])) )
elif opts.condor:
	os.system("%s/tools/submit_condor_jobs_file cmd_multipleregress_randomise_%s_%d" % (SCRIPTPATH,opts.surface[0],currentTime) )
elif opts.fslsub:
	os.system("${FSLDIR}/bin/fsl_sub -t cmd_multipleregress_randomise_%s_%d" % (opts.surface[0],currentTime) )

print "Run: ${SCRIPTPATH}/tools/calculate_fweP_vertex.py to calculate (1-P[FWE]) image (after randomisation is finished)."
