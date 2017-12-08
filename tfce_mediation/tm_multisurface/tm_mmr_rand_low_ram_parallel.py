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

	ap.add_argument("-sn", "--surfacenumber",
		nargs=1, 
		type=int,
		metavar=('INT'))
	ap.add_argument("-pr", "--permutationrange",
		nargs=2, 
		type=int,
		metavar=('INT','INT'))
	return ap


def run(opts):
	opts = np.load("tmi_temp/opts.npy")


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
