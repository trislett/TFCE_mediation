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
import sys
import argparse 

import numpy as np
import nibabel as nib

from scipy.stats import linregress

from tfce_mediation.stats import resid_covars, calc_beta_se
from tfce_mediation.tfce import Surface
from tfce_mediation.func import write_voxelStat_img, create_adjac_voxel, calc_sobelz

DESCRIPTION = "Voxel-wise mediation with TFCE"

def getArgumentParser(parser = argparse.ArgumentParser(description = DESCRIPTION)):
  parser.add_argument("-i", "--input", 
    nargs = 3, 
    help = "[predictor file] [covariate file] [dependent file]", 
    metavar = ('*.csv', '*.csv', '*.csv'), 
    required = True)
  parser.add_argument("-m", "--medtype", 
    nargs = 1, 
    help = "Voxel-wise mediation type", 
    choices = ['I','M','Y'], 
    required=True)
  parser.add_argument("-t", "--tfce", 
    help="H E Connectivity. Default is 2 1 26.", 
    nargs = 3, 
    default = [2, 1, 26], 
    metavar = ('H', 'E', '[6 or 26]'))
  return parser

def run(opts):
  arg_predictor = opts.input[0]
  arg_covars = opts.input[1]
  arg_depend = opts.input[2]
  medtype = opts.medtype[0]

  if not os.path.exists("python_temp"):
    print "python_temp missing!"

  #load variables
  raw_nonzero = np.load('python_temp/raw_nonzero.npy')
  n = raw_nonzero.shape[1]
  header_mask = np.load('python_temp/header_mask.npy')
  affine_mask = np.load('python_temp/affine_mask.npy')
  data_mask = np.load('python_temp/data_mask.npy')
  data_index = data_mask>0.99
  num_voxel = np.load('python_temp/num_voxel.npy')
  pred_x = np.genfromtxt(arg_predictor, delimiter=",")
  covars = np.genfromtxt(arg_covars, delimiter=",")
  depend_y = np.genfromtxt(arg_depend, delimiter=",")

  #TFCE
  adjac = create_adjac_voxel(data_index,data_mask,num_voxel,dirtype=opts.tfce[2])
  calcTFCE = Surface(float(opts.tfce[0]), float(opts.tfce[1]), adjac) # i.e. default: H=2, E=2, 26 neighbour connectivity

  #step1
  x_covars = np.column_stack([np.ones(n),covars])
  y = resid_covars(x_covars,raw_nonzero)

  #save
  np.save('python_temp/pred_x',pred_x)
  np.save('python_temp/covars',covars)
  np.save('python_temp/depend_y',depend_y)
  np.save('python_temp/adjac',adjac)
  np.save('python_temp/medtype',medtype)
  np.save('python_temp/optstfce', tfce)
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

