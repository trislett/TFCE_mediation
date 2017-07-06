#!/usr/bin/env python

#    convert-surface
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
import numpy as np
import nibabel as nib
import argparse as ap
import matplotlib.pyplot as plt
from tfce_mediation.pyfunc import convert_mni_object, convert_fs, convert_gifti, convert_ply, convert_fslabel, save_waveform, save_stl, save_fs, save_ply, convert_redtoyellow, convert_bluetolightblue, convert_mpl_colormaps, convert_fsannot

DESCRIPTION = """
Conversion of surfaces (freesurfer, gifti *.gii, mni *.obj, ply *.ply) to freesurfer surface or other objects (Waveform *obj, STereoLithography *stl, Polygon File Format *ply) for analysis with TFCE_mediation. *mgh files can also be imported and converted to PLY files.
"""

#arguments parser
def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):
	#input type
	igroup = ap.add_mutually_exclusive_group(required=True)
	igroup.add_argument("-i_fs", "--inputfreesurfer",
		help="Input a freesurfer surface (e.g., -i_fs $SUBJECTS_DIR/fsaverage/surf/lh.midthickness)", 
		nargs=1, 
		metavar=('*'))
	igroup.add_argument("-i_gifti", "--inputgifti",
		help="Input a gifti surface file (e.g., --i_gifti average.surf.gii)", 
		nargs=1, 
		metavar=('*.surf.gii'))
	igroup.add_argument("-i_mni", "--inputmniobj",
		help="Input a MNI object file (e.g., --i_mni l_hemi.obj)", 
		nargs=1, 
		metavar=('*.obj'))
	igroup.add_argument("-i_ply", "--inputply",
		help="Input a MNI object file (e.g., --i_ply l_hemi.ply). Note, vertex colors will be stripped.", 
		nargs=1, 
		metavar=('*.ply'))
	ogroup = ap.add_mutually_exclusive_group(required=True)
	ogroup.add_argument("-o_fs", "--outputfreesurfer",
		help="Output file name for freesurfer surface (e.g., -o_fs lh.32k.midthickness)", 
		nargs=1, 
		metavar=('*'))
	ogroup.add_argument("-o_obj", "--outputwaveform",
		help="Output file name for waveform object file for visualization with blender (or any other 3D viewer). This is NOT an MNI object file.", 
		nargs=1, 
		metavar=('*'))
	ogroup.add_argument("-o_stl", "--outputstl",
		help="Output file name for STereoLithography (STL) object file for visualization with blender (or any other 3D viewer).", 
		nargs=1, 
		metavar=('*'))
	ogroup.add_argument("-o_ply", "--outputply",
		help="Output file name for Polygon File Format (PYL) object file for visualization with blender (or any other 3D viewer).", 
		nargs=1, 
		metavar=('*'))

	ap.add_argument("-p", "--paintsurface",
		help="Projects surface file onto a ply mesh for visualization of results using a 3D viewer. Must be used with -o_ply option. Input the surface file (*.mgh), the sigificance threshold (low and high), and either: red-yellow (r_y), blue-lightblue (b_lb) or any matplotlib colorschemes (https://matplotlib.org/examples/color/colormaps_reference.html). Note, thresholds must be postive. e.g., -p image.mgh 0.95 1 r_y", 
		nargs=4, 
		metavar=('*.mgh','float','float', 'colormap'))
	ap.add_argument("-s", "--paintsecondsurface",
		help="Projects a second surface file onto a ply mesh for visualization of resutls using a 3D viewer. Must be used with -o_ply and -p options. Input the surface file (*.mgh), the sigificance threshold (low and high), and either: red-yellow (r_y), blue-lightblue (b_lb) or any matplotlib colorschemes (https://matplotlib.org/examples/color/colormaps_reference.html). Note, thresholds must be postive. e.g., -s negimage.mgh 0.95 1 b_lb", 
		nargs=4, 
		metavar=('*.mgh','float','float', 'colormap'))

	ap.add_argument("-l", "--paintfslabel",
		help="Projects freesurface label file onto a ply mesh for visualization of resutls using a 3D viewer. Must be used with -o_ply option. Input the label (*.label or *.label-????) and either: red-yellow (r_y), blue-lightblue (b_lb) or any matplotlib colorschemes (https://matplotlib.org/examples/color/colormaps_reference.html). More than one label can be included. e.g. -l label1.label rainbow label2.label Reds", 
		nargs='+', 
		metavar=('*.label colormap'))
	ap.add_argument("-a", "--paintfsannot",
		help="Projects freesurface annotation file onto a ply mesh for visualization of resutls using a 3D viewer. Must be used with -o_ply option. The legend is outputed", 
		nargs=1, 
		metavar=('*.annot'))


	return ap

def run(opts):
	#input
	if opts.inputfreesurfer:
		v,f = convert_fs(str(opts.inputfreesurfer[0]))
	if opts.inputgifti:
		v,f = convert_gifti(str(opts.inputgifti[0]))
	if opts.inputmniobj:
		v,f = convert_mni_object(str(opts.inputmniobj[0]))
	if opts.inputply:
		v,f,_ = convert_ply(str(opts.inputply[0]))
	#output
	if opts.outputfreesurfer:
		save_fs(v,f, opts.outputfreesurfer[0])
	if opts.outputwaveform:
		save_waveform(v,f, opts.outputwaveform[0])
	if opts.outputstl:
		save_stl(v,f, opts.outputstl[0])
	if opts.outputply:
		# get the matplotlib colormaps
		colormaps = np.array(plt.colormaps(),dtype=np.str)
		if opts.paintsurface:
			img = nib.load(opts.paintsurface[0])
			img_data = img.get_data()
			if img_data.ndim > 3:
				print "Error: input file can only contain one subject"
				quit()
			img_data = img_data[:,0,0]
			if (str(opts.paintsurface[3]) == 'r_y') or (str(opts.paintsurface[3]) == 'red-yellow'):
				out_color_array = convert_redtoyellow(np.array(( float(opts.paintsurface[1]),float(opts.paintsurface[2]) )), img_data)
			elif (str(opts.paintsurface[3]) == 'b_lb') or (str(opts.paintsurface[3]) == 'blue-lightblue'):
				out_color_array = convert_bluetolightblue(np.array(( float(opts.paintsurface[1]),float(opts.paintsurface[2]) )), img_data)
			elif np.any(colormaps == str(opts.paintsurface[3])):
				out_color_array = convert_mpl_colormaps(np.array(( float(opts.paintsurface[1]),float(opts.paintsurface[2]) )), img_data, str(opts.paintsurface[3]))
			else:
				print "Colour scheme %s does not exist" % str(opts.paintsurface[3])
				quit()
			if opts.paintsecondsurface:
				img = nib.load(opts.paintsecondsurface[0])
				img_data = img.get_data()
				if img_data.ndim > 3:
					print "Error: input file can only contain one subject"
					quit()
				img_data = img_data[:,0,0]
				index = img_data > float(opts.paintsecondsurface[1])
				if (str(opts.paintsecondsurface[3]) == 'r_y') or (str(opts.paintsecondsurface[3]) == 'red-yellow'):
					out_color_array2 = convert_redtoyellow(np.array(( float(opts.paintsecondsurface[1]),float(opts.paintsecondsurface[2]) )), img_data)
				elif (str(opts.paintsecondsurface[3]) == 'b_lb') or (str(opts.paintsecondsurface[3]) == 'blue-lightblue'):
					out_color_array2 = convert_bluetolightblue(np.array(( float(opts.paintsecondsurface[1]),float(opts.paintsecondsurface[2]) )), img_data)
				elif np.any(colormaps == str(opts.paintsecondsurface[3])):
					out_color_array2 = convert_mpl_colormaps(np.array(( float(opts.paintsecondsurface[1]),float(opts.paintsecondsurface[2]) )), img_data, str(opts.paintsecondsurface[3]))
				else:
					print "Error: colour scheme %s does not exist" % str(opts.paintsecondsurface[3])
					quit()
				out_color_array[index,:] = out_color_array2[index,:]
			save_ply(v,f, opts.outputply[0], out_color_array)
		elif opts.paintfslabel:
			labelpairs = len(opts.paintfslabel)
			numlabels = labelpairs /2
			counter=0
			for i in range(int(numlabels)):
				v_id, _ , v_value = convert_fslabel(str(opts.paintfslabel[counter]))
				counter+=1
				icolormap = str(opts.paintfslabel[counter])
				counter+=1
				max_value = v_value.max()
				if (str(icolormap) == 'r_y') or (str(icolormap) == 'red-yellow'):
					out_color_label = convert_redtoyellow(np.array(( float(0),float(max_value) )), v_value)
				elif (str(icolormap) == 'b_lb') or (str(icolormap) == 'blue-lightblue'):
					out_color_label = convert_bluetolightblue(np.array(( float(0),float(max_value) )), v_value)
				elif np.any(colormaps == str(icolormap)):
					out_color_label = convert_mpl_colormaps(np.array(( float(0),float(max_value) )), v_value, str(icolormap))
				else:
					print "Error: colour scheme %s does not exist" % str(icolormap)
					quit()
				if counter == 2: # dirty method
					baseColour=[227,218,201] # bone
					out_color_array = np.zeros((v.shape[0],3))
					out_color_array[:,:]=baseColour
				out_color_array[v_id]=out_color_label
			save_ply(v,f, opts.outputply[0], out_color_array)
		elif opts.paintfsannot:
			out_color_array = convert_fsannot(opts.paintfsannot[0])
			save_ply(v,f, opts.outputply[0], out_color_array)
		else:
			save_ply(v,f, opts.outputply[0])

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)


