#!/usr/bin/env python

import numpy as np
import nibabel as nib
import argparse as ap

from tfce_mediation.pyfunc import loadmgh

def writeCSV(base,tail,data):
	np.savetxt(("%s_%s.csv" % (base,tail)), data, delimiter=",", fmt='%10.8f')

DESCRIPTION = "Converts a text file of surface values (e.g. cortical thickness values at each vertices) to a freesurfer 'surface', and vice versa. n.b., correct order of the vertices is assumed."

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument("-i", "--inputtxt",
		nargs=1,
		help="input a text file surface values (i.e., a list of cortical thickness values)")
	group.add_argument("-s", "--inputsurface",
		nargs=1,
		help="input a surface", 
		metavar=('*.mgh'))
	ap.add_argument("-t", "--transpose",
		action='store_true',
		help="Transpose input file.")
	return ap

def run(opts):
	if opts.inputtxt:
		txtname = opts.inputtxt[0]
		surfVals = np.loadtxt(txtname, dtype=np.float, delimiter=',')
		if opts.transpose:
			print "Transposing array"
			surfVals = surfVals.T
		numVert = len(surfVals)
		print "Reading in %d subjects and %d vertices. If this is incorrect re-run the script is the --transpose option." % (surfVals.shape[1], surfVals.shape[0])
		if surfVals.ndim == 1:
			outsurf = np.zeros((numVert,1,1))
			outsurf[:,0,0] = surfVals
		if surfVals.ndim == 2:
			outsurf = np.zeros((numVert,1,1,surfVals.shape[1]))
			outsurf[:,0,0,:] = surfVals
		nib.save(nib.freesurfer.mghformat.MGHImage(outsurf.astype(np.float32, order = "C"),affine=None),'%s.mgh' % txtname)
	if opts.inputsurface:
		surfname = opts.inputsurface[0]
		img, imgdata = loadmgh(surfname)
		if imgdata.ndim==3:
			imgdata = imgdata[:,0,0]
		if imgdata.ndim==4:
			imgdata = imgdata[:,0,0,:]
		np.savetxt(("%s.csv" % surfname), imgdata, delimiter=",", fmt='%10.8f')
if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
