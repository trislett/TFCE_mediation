#!/usr/bin/python

import os
import argparse as ap

DESCRIPTION = "Quick python wrapper for freeview"

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i", "--image", 
		help="image_?h.mgh OR image_lh.mgh image_rh.mgh", 
		nargs='+', 
		metavar=('*.mgh'), 
		required=True)
	ap.add_argument("-s", "--surface", 
		help="fsaverage surface (e.g., pial, inflated)", 
		default=['midthickness'], 
		nargs=1)
	ap.add_argument("--hemi", help="Hemisphere (default is lh)", 
		choices=['lh', 'rh'], 
		default=['lh'])
	ap.add_argument("-l", "--lower", 
		help="Lower threshold (default is 0.95)", 
		default=['0.95'], 
		nargs=1)
	ap.add_argument("-u", "--upper", 
		help="Upper threshold (default is 1)", 
		default=['1.00'], 
		nargs=1)
	return ap

def run(opts):
	if len(opts.image)==2:
		sysout = 'freeview -f %s/fsaverage/surf/lh.%s:overlay=%s:overlay_threshold=%s,%s -f %s/fsaverage/surf/rh.%s:overlay=%s:overlay_threshold=%s,%s' %(os.environ.get('SUBJECTS_DIR'),opts.surface[0],opts.image[0],opts.lower[0],opts.upper[0],os.environ.get('SUBJECTS_DIR'),opts.surface[0],opts.image[1],opts.lower[0],opts.upper[0])
	else:
		sysout = 'freeview -f %s/fsaverage/surf/%sh.%s:overlay=%s:overlay_threshold=%s,%s' %(os.environ.get('SUBJECTS_DIR'),opts.hemi[0],opts.surface[0],opts.image[0],opts.lower[0],opts.upper[0])
	os.system(sysout)

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
