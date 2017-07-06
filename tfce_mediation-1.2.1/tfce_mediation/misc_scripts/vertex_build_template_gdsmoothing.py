#!/usr/bin/env python

import os
import argparse
import numpy as np
import time
import datetime

DESCRIPTION = "Python pipe to build Freesurfer template surface. $SUBJECT_DIR and $TM_ADDONS must be declared (e.g., export SUBJECT_DIR=/path/to/freesurfer/subjects)"

def getArgumentParser(parser = argparse.ArgumentParser(description = DESCRIPTION)):
	parser.add_argument("-i", "--input", 
		nargs=2, 
		help="[subject_list] [surface]", 
		metavar=('*.csv', 'area or thickness'),
		required=True)
#	parser.add_argument("-f", "--fwhm", 
#		nargs='?', 
#		help="Optional. Specify FWHM smoothing (other than 0,3,and 10). E.g., -f 2]", 
#		metavar=('INT'))
	parser.add_argument("-n", "--numcores", 
		help="Number of cores to use for parallel processing", 
		nargs=1,
		metavar=('INT'))
	parser.add_argument("-f", "--geodesicfwhm", 
		help="Geodesic FWHM value", 
		type=float,
		nargs=1, 
		default=[3.0],
		metavar=('FLOAT'))
	return parser

def run(opts):

#	if subjects.dtype.kind in np.typecodes['AllInteger']:
#		subjects = np.array(map(str, subjects))
#	surface = str(opts.input[1])
	subject_list = opts.input[0]
	subjects = np.genfromtxt(str(subject_list), delimiter=",", dtype=str)
	surface = opts.input[1]
	numcore = int(opts.numcores[0])
	fwhm = float(opts.geodesicfwhm[0])
	currenttime = time.time()
	timestamp = datetime.datetime.fromtimestamp(currenttime).strftime('%Y_%m_%d_%H%M%S')
	tempdir = "temp_%s_%s" % (surface,timestamp)
	cmd_reg =  "cmd_reg_%s_%s" % (surface, timestamp)
	cmd_smooth =  "cmd_smooth_%s_%s" % (surface, timestamp)

	os.system('echo "Current SUBJECTS_DIR is: " $SUBJECTS_DIR;')
	os.system("""
		mkdir %s
		for hemi in lh rh; do
			for i in $(cat %s); do
				echo mri_surf2surf --srcsubject ${i} --srchemi ${hemi} --srcsurfreg sphere.reg --trgsubject fsaverage --trghemi ${hemi} --trgsurfreg sphere.reg --tval ./%s/${hemi}.${i}.%s.00.mgh --sval ${SUBJECTS_DIR}/${i}/surf/${hemi}.%s --jac --sfmt curv --noreshape --cortex
			done >> %s/%s
		done
		cat %s/%s | parallel -j %d;
		""" % (tempdir,subject_list, tempdir,surface,surface,tempdir,cmd_reg, tempdir,cmd_reg, numcore) )
	if surface == 'area':
		print "Merging images for Box-Cox correction"
		os.system("""for hemi in lh rh; do 
			tm_tools merge-images --vertex -o ${hemi}.all.%s.00.mgh -i %s/${hemi}*.mgh
			done""" % (surface, tempdir) )
		os.system("rm -rf %s" % tempdir)
		print "Performing Box-Cox correction"
		os.system("""for hemi in lh rh; do
			tm_tools vertex-box-cox-transform -i ${hemi}.all.%s.00.mgh %d --nosmoothing
			done""" % (surface, numcore) )
		os.mkdir(tempdir)
		os.system("cp ??.all.%s.00.boxcox.mgh %s/" % (surface,tempdir))
		os.chdir('%s/' % tempdir)
		os.system("""for hemi in lh rh; do
			tm_maths --vertex ${hemi}.all.%s.00.boxcox.mgh -o ${hemi}.all.%s.00.boxcox.mgh --split
			rm ${hemi}.all.%s.00.boxcox.mgh
			done""" % (surface, surface, surface) )
		print "Performing Geodesic smoothing with FWHM = %1.1f " % fwhm
		os.system("""
		for hemi in lh rh; do
			for i in img*${hemi}*; do 
				echo tm_tools geodesic-fwhm -i $i -o smoothed_${i} --hemi ${hemi} -f %1.4f -d ${TM_ADDONS}/${hemi}_8.0mm_fwhm_distances.npy
			done >> %s
		done
		cat %s | parallel -j %d;
		""" % (fwhm,cmd_smooth,cmd_smooth,numcore) )
		os.chdir('../')
		os.system("""for hemi in lh rh; do 
			tm_tools merge-images --vertex -o ${hemi}.all.%s_boxcox.03B.mgh -i %s/smoothed*img*${hemi}*.mgh
			done""" % (surface, tempdir))
		os.system("rm -rf %s" % tempdir)
	else:
		print "Performing Geodesic smoothing with FWHM = %1.1f " % fwhm
		os.chdir('%s/' % tempdir)
		os.system("""
		for hemi in lh rh; do
			for i in ${hemi}*.mgh; do
				echo tm_tools geodesic-fwhm -i $i -o smoothed_${i} --hemi ${hemi} -f %1.4f -d ${TM_ADDONS}/${hemi}_8.0mm_fwhm_distances.npy
			done >> %s
		done
		cat %s | parallel -j %d;
		""" % (fwhm,cmd_smooth,cmd_smooth,numcore) )
		os.chdir('../')
		os.system("""for hemi in lh rh; do
			tm_tools merge-images --vertex -o ${hemi}.all.%s.00.mgh -i %s/${hemi}*00.mgh 
			tm_tools merge-images --vertex -o ${hemi}.all.%s.03B.mgh -i %s/smoothed*${hemi}*00.mgh
			done""" % (surface, tempdir))
		os.system("rm -rf %s" % tempdir)
if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)

