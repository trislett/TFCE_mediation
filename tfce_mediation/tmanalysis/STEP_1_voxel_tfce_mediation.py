#!/usr/bin/env python

#    Voxel-wise mediation with TFCE
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
import argparse as ap

import numpy as np

from tfce_mediation.cynumstats import resid_covars
from tfce_mediation.tfce import CreateAdjSet
from tfce_mediation.pyfunc import write_voxelStat_img, create_adjac_voxel, calc_sobelz

DESCRIPTION = "Voxel-wise mediation with TFCE"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i", "--input", 
		nargs = 2, 
		help = "[predictor file] [dependent file]", 
		metavar = ('*.csv', '*.csv'), 
		required = True)
	ap.add_argument("-c", "--covariates", 
		nargs = 1, 
		help = "[covariate file]", 
		metavar = ('*.csv'))
	ap.add_argument("-m", "--medtype", 
		nargs = 1, 
		help = "Voxel-wise mediation type", 
		choices = ['I','M','Y'], 
		required=True)
	ap.add_argument("-t", "--tfce", 
		help="H E Connectivity. Default is 2 1 26.", 
		nargs = 3, 
		default = [2, 1, 26], 
		metavar = ('H', 'E', '[6 or 26]'))
	return ap

def run(opts):
	arg_predictor = opts.input[0]
	arg_depend = opts.input[1]
	medtype = opts.medtype[0]

	if not os.path.exists("python_temp"):
		print "python_temp missing!"

	#load variables
	raw_nonzero = np.load('python_temp/raw_nonzero.npy')
	n = raw_nonzero.shape[1]
	affine_mask = np.load('python_temp/affine_mask.npy')
	data_mask = np.load('python_temp/data_mask.npy')
	data_index = data_mask>0.99
	num_voxel = np.load('python_temp/num_voxel.npy')
	pred_x = np.genfromtxt(arg_predictor, delimiter=",")
	depend_y = np.genfromtxt(arg_depend, delimiter=",")

	#TFCE
	adjac = create_adjac_voxel(data_index,data_mask,num_voxel,dirtype=opts.tfce[2])
	calcTFCE = CreateAdjSet(float(opts.tfce[0]), float(opts.tfce[1]), adjac) # i.e. default: H=2, E=2, 26 neighbour connectivity

	#step1
	if opts.covariates:
		arg_covars = opts.covariates[0]
		covars = np.genfromtxt(arg_covars, delimiter=",")
		x_covars = np.column_stack([np.ones(n),covars])
		y = resid_covars(x_covars,raw_nonzero)
	else:
		y = raw_nonzero.T

	#save
	np.save('python_temp/pred_x',pred_x)
	np.save('python_temp/depend_y',depend_y)
	np.save('python_temp/adjac',adjac)
	np.save('python_temp/medtype',medtype)
	np.save('python_temp/optstfce', opts.tfce)
	np.save('python_temp/raw_nonzero_corr',y.T.astype(np.float32, order = "C"))

	#step2 mediation
	SobelZ = calc_sobelz(medtype, pred_x, depend_y, y, n, num_voxel)

	#write TFCE images
	if not os.path.exists("output_med_%s" % medtype):
		os.mkdir("output_med_%s" % medtype)
	os.chdir("output_med_%s" % medtype)
	write_voxelStat_img('SobelZ_%s' % medtype, SobelZ, data_mask, data_index, affine_mask, calcTFCE)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
