#!/usr/bin/env python

#    read tmi image file
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




DESCRIPTION = "Converts a text file of surface values (e.g. cortical thickness values at each vertices) to a freesurfer 'surface', and vice versa. n.b., correct order of the vertices is assumed."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):

	ap.add_argument("-i", "--inputtmi",
		nargs=1,
		help="input a tfce_mediation image (*.tmi) binary or ascii file.",
		metavar=('*.tmi'),
		required=True)
	ap.add_argument("-o", "--outputbasicinfo",
		help="Output the basic infomation.",
		action='store_true')
	ap.add_argument("--fileformat",
		help="Output the file format.",
		action='store_true')
	ap.add_argument("--headersize",
		help="Output the header size in number of lines and bytes.",
		action='store_true')
	ap.add_argument("-oh", "--outputhistory",
		help="Output the history.",
		action='store_true')
	ap.add_argument("-om", "--outputmaskinfo",
		help="Output mask(s) information.",
		action='store_true')
	ap.add_argument("-os", "--outputshapeinfo",
		help="Output shape object(s) information.",
		action='store_true')
	return ap

def run(opts):
	tm_file = opts.inputtmi[0]
	# getfilesize
	filesize = os.stat(tm_file).st_size
	# declare variables
	element = []
	element_dtype = []
	element_nbyte = []
	element_nmasked = []
	datashape = []
	maskshape = []
	maskname = []
	vertexshape = []
	faceshape = []
	surfname = []
	affineshape = []
	adjlength = []
	maskcounter=0
	vertexcounter=0
	affinecounter=0
	adjacencycounter=0
	tmi_history = []

	# read first line
	obj = open(tm_file)
	reader = obj.readline().strip().split()
	firstword=reader[0]
	if firstword != 'tmi':
		print "Error: not a TFCE_mediation image."
		exit()
	reader = obj.readline().strip().split()
	firstword=reader[0]
	if firstword != 'format':
		print "Error: unknown reading file format"
		exit()
	else:
		tm_filetype = reader[1]
		tmi_version = reader[2]

	linecounter=0
	while firstword != 'end_header':
		linecounter+=1
		reader = obj.readline().strip().split()
		firstword=reader[0]
		if firstword=='element':
			element.append((reader[1]))
		if firstword=='dtype':
			element_dtype.append((reader[1]))
		if firstword=='nbytes':
			element_nbyte.append((reader[1]))
		if firstword=='datashape':
			datashape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword=='nmasked':
			element_nmasked.append(( int(reader[1]) ))
			maskcounter+=1 #dirty
		if firstword=='maskshape':
			maskshape.append(np.array((reader[1], reader[2], reader[3])).astype(np.int))
		if firstword=='maskname':
			maskname.append((reader[1]))
		if firstword=='affineshape':
			affineshape.append(np.array((reader[1], reader[2])).astype(np.int))
			affinecounter+=1 #dirty
		if firstword=='vertexshape':
			vertexshape.append(np.array((reader[1], reader[2])).astype(np.int))
			vertexcounter+=1 #dirty
		if firstword=='faceshape':
			faceshape.append(np.array((reader[1], reader[2])).astype(np.int))
		if firstword=='surfname':
			surfname.append((reader[1]))
		if firstword=='adjlength':
			adjlength.append(np.array(reader[1]).astype(np.int))
			adjacencycounter+=1 #dirty
		if firstword=='history':
			tmi_history.append(str(' '.join(reader)))
	# skip header
	position = filesize
	for i in range(len(element_nbyte)):
		position-=int(element_nbyte[i])

	if opts.outputbasicinfo:
		print "--- Basic Information ---"
		for i in range(len(element)):
			if element[i]=='data_array':
				print ("Data array shape: %dx%d [masked data length by # images (subjects)]" % (datashape[0][0],datashape[0][1]))
		print ("Number of masks: %d" % maskcounter)
		print ("Number of affines: %d" % affinecounter)
		print ("Number of surfaces: %d" % vertexcounter)
		print ("Number of adjacency sets: %d" % adjacencycounter)
		print ("")

	if opts.fileformat:
			print "--- File Information ---"
			print "File type: %s" % tm_filetype
			print "Version: %s\n" % tmi_version

	if opts.headersize:
		print "--- Header Information ---"
		if tm_filetype == 'ascii':
			print "Header size: %d lines\n" % (linecounter+2)
		else:
			print "Header size: %d bytes, %d lines\n" % (position,linecounter+2)

	if opts.outputhistory:
		print "--- History ---"
		for i in range(len(tmi_history)):
			print "Time-point %d" % i
			line = tmi_history[i].split(' ')
			print ("Date: %s-%s-%s %s:%s:%s" % (line[2][6:8],line[2][4:6],line[2][0:4], line[2][8:10], line[2][10:12], line[2][12:14]) )
			if line[1]=='mode_add':
				print "Elements added:"
			elif line[1]=='mode_sub':
				print "Elements removed:"
			else:
				print "Error: mode is not understood"
			print "Number of masks: %s" % line[4]
			print "Number of affines: %s" % line[5]
			print "Number of surfaces: %s" % line[6]
			print "Number of adjacency sets: %s\n" % line[7]

	if opts.outputmaskinfo:
		print "--- Mask Information ---"
		for i in range(len(maskname)):
			print "Mask element %d" % i
			print "Mask name: %s" % maskname[i]
			print "Mask size: %s" % element_nmasked[i]
			print ("Mask shape: %s %s %s\n" % (maskshape[i][0],maskshape[i][1],maskshape[i][2]) )

	if opts.outputshapeinfo:
		print "--- Shape Information ---"
		for i in range(len(surfname)):
			print "Shape element %d" % i
			print "Shape name: %s" % surfname[i]
			print ("Vertices: %s %s" % (vertexshape[i][0],vertexshape[i][1]) )
			print ("Faces: %s %s\n" % (faceshape[i][0],faceshape[i][1]) )

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
