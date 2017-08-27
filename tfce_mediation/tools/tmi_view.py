#!/usr/bin/env python

#    view *.tmi images for TFCE_mediation
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

from __future__ import division
import os
import numpy as np
import argparse as ap

from tfce_mediation.tm_io import read_tm_filetype
from tfce_mediation.tm_func import print_tmi_history
from tfce_mediation.pyfunc import convert_voxel
# hopefully safer loafing of mayavi
try:
	from mayavi import mlab
except:
	if 'QT_API' not in os.environ: 
		os.environ['QT_API'] = 'pyside'
	try: 
		from mayavi import mlab
	except:
		os.environ['QT_API'] = 'pyqt'
		from mayavi import mlab

DESCRIPTION = """
Display surface statistics from a tmi file.

Note: this script relies on Mayavi. If you use, please cite:
Ramachandran, P. and Varoquaux, G., `Mayavi: 3D Visualization of Scientific Data` IEEE Computing in Science & Engineering, 13 (2), pp. 40-51 (2011)
"""

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):

	ap.add_argument("-i_tmi", "--inputtmi",
		help="Input the *.tmi file containing the statistics to view.",
		nargs=1, 
		metavar='*.tmi',
		required=True)
	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument("-oh", "--history",
		help="Output tmi file history and exits.", 
		action='store_true')
	group.add_argument("-d", "--display",
		help="Select which object to display. The mask, surface, contrast must be entered as integers (check values with -oh). Multiple objects can be displayed. The input must be divisible by three.",
		nargs = '+',
		type = int,
		metavar = 'int')
	group.add_argument("--onlyvoxel",
		help="display only voxel images", 
		action='store_true')
	# optional
	ap.add_argument("-dm", "--displayvoxelmask",
		help = "Display a voxel surface.", 
		nargs = '+',
		type = int,
		metavar = 'int')
	ap.add_argument("-lut", "--lookuptable",
		help = "Set the color map to display.", 
		choices = ('red-yellow','blue-lightblue','r_y','b_lb'),
		type = str,
		default = ['r_y'],
		nargs = 1)
	ap.add_argument("-t", "--thresholds",
		help = "Set upper and lower thresholds", 
		default=[.95,1],
		type = float,
		nargs = 2)
	ap.add_argument("-a", "--alpha",
		help = "Set alpha [0 to 255]", 
		default=[255],
		type = int,
		nargs = 1)
	return ap

def run(opts):
	# load tmi
	_, image_array, masking_array, maskname_array, affine_array, vertex_array, face_array, surfname, _, tmi_history, columnids = read_tm_filetype(opts.inputtmi[0], verbose=False)
	if opts.history:
		print_tmi_history(tmi_history, maskname_array, surfname, num_con = image_array[0].shape[1], contrast_names = columnids)
		quit()

	# get the positions of masked data in image_array
	pointer = 0
	position_array = [0]
	for i in range(len(masking_array)):
		pointer += len(masking_array[i][masking_array[i]==True])
		position_array.append(pointer)
	del pointer

	# make custom look-up table
	if (str(opts.lookuptable[0]) == 'r_y') or (str(opts.lookuptable[0]) == 'red-yellow'):
		cmap_array = np.array(( (np.ones(256)*255), np.linspace(0,255,256), np.zeros(256), np.ones(256)*255.0)).T
	elif (str(opts.lookuptable[0]) == 'b_lb') or (str(opts.lookuptable[0]) == 'blue-lightblue'):
		cmap_array = np.array(( np.zeros(256), np.linspace(0,255,256), (np.ones(256)*255), np.ones(256)*255.0)).T
	else:
		pass # change later
	cmap_array[0] = [227,218,201,opts.alpha[0]] # set lowest threshold to bone

	if opts.display:
		if len(opts.display) % 3 != 0:
			print "Error"
			quit()
		num_obj = int(len(opts.display) / 3)

		# display the surfaces
		for i in range(num_obj):
			c_mask = opts.display[(0 + int(i*3))]
			c_surf = opts.display[(1 + int(i*3))]
			c_contrast = opts.display[(2 + int(i*3))]

			start = position_array[c_mask]
			end = position_array[c_mask+1]

			mask = masking_array[c_mask]
			scalar_data = np.zeros((mask.shape[0]))
			scalar_data[mask[:,0,0]] = image_array[0][start:end,c_contrast]

			surf = mlab.triangular_mesh(vertex_array[c_surf][:,0],vertex_array[c_surf][:,1],vertex_array[c_surf][:,2],face_array[c_surf], scalars=scalar_data, vmin=opts.thresholds[0], vmax=opts.thresholds[1], name = maskname_array[c_mask])
			surf.module_manager.scalar_lut_manager.lut.table = cmap_array

	if opts.displayvoxelmask:

		cmap_array[0] = [227,218,201,0]

		if len(opts.displayvoxelmask) % 2 != 0:
			print "Error"
			quit()

		for j in range(int(len(opts.displayvoxelmask)/2)):
			c_mask = opts.displayvoxelmask[0 + int(j*2)]
			c_contrast = opts.displayvoxelmask[1 + int(j*2)]

			start = position_array[c_mask]
			end = position_array[c_mask+1]

			mask = masking_array[c_mask]
			scalar_data = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2]))
			scalar_data[mask] = image_array[0][start:end,c_contrast]
			v, f, values = convert_voxel(scalar_data, affine = affine_array[c_mask], absthreshold = opts.thresholds[0])
			surf = mlab.triangular_mesh(v[:,0],v[:,1],v[:,2],f, scalars=values, scale_factor=0, vmin=opts.thresholds[0], vmax=opts.thresholds[1], name = maskname_array[c_mask], transparent=True)
			surf.module_manager.scalar_lut_manager.lut.table = cmap_array

	mlab.show()

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)

