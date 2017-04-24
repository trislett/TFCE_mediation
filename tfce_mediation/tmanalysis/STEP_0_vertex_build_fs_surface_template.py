#!/usr/bin/env python

#    Load mean_FA_skeleton_mask and all_FA_skeletonised into tfce_mediation
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
import argparse
import numpy as np
import time
import datetime

DESCRIPTION = "Python pipe to build Freesurfer template surface using mri_surf2surf. $SUBJECT_DIR must be declared (e.g., export SUBJECT_DIR=/path/to/freesurfer/subjects)"

def getArgumentParser(parser = argparse.ArgumentParser(description = DESCRIPTION)):
	parser.add_argument("-i", "--input", 
		nargs=2, 
		help="[subject_list] [surface]", 
		metavar=('*.csv', 'area or thickness'),
		required=True)
	parser.add_argument("-f", "--fwhm", 
		nargs='+', 
		help="Optional. Specify FWHM smoothing other than 3mm. Multiple smoothing values can be specificed, E.g.,, for  FWHM = 4mm and 10mm ( -f 4 10 )",
		metavar=('INT'))
	parser.add_argument("-p", "--parallel", 
		nargs=1,
		type=int,
		help="Optional (recommended). Use GNU parallel processing with entering the number of cores (e.g., -p 8)", 
		metavar=('INT'))
	parser.add_argument("-k", "--noclean", 
		help="Keep temporary files", 
		action='store_true')
	return parser

def run(opts):
	subject_list = opts.input[0]
	subjects = np.genfromtxt(str(subject_list), delimiter=",", dtype=str)
#	if subjects.dtype.kind in np.typecodes['AllInteger']:
#		subjects = np.array(map(str, subjects))
	surface = str(opts.input[1])
	if opts.parallel:
		numcore = opts.parallel[0]

	currenttime = time.time()
	timestamp = datetime.datetime.fromtimestamp(currenttime).strftime('%Y_%m_%d_%H%M%S')
	tempdir = "temp_%s_%s" % (surface,timestamp)
	cmd_reg =  "cmd_reg_%s_%s" % (surface, timestamp)
	cmd_smooth =  "cmd_smooth_%s_%s" % (surface, timestamp)

	os.system('echo "Current SUBJECTS_DIR is: " $SUBJECTS_DIR;')
#	longname='{1}{0}'.format(' --s '.join(subjects), '--s ')
#	with open("longname", "w") as text_file:
#		text_file.write("%s" % longname)

	os.system("""
		mkdir %s;
		for hemi in lh rh; do
			for i in $(cat %s); do
				echo $FREESURFER_HOME/bin/mri_surf2surf --srcsubject ${i} --srchemi ${hemi} --srcsurfreg sphere.reg --trgsubject fsaverage --trghemi ${hemi} --trgsurfreg sphere.reg --tval ./%s/${hemi}.${i}.%s.00.mgh --sval ${SUBJECTS_DIR}/${i}/surf/${hemi}.%s --jac --sfmt curv --noreshape --cortex
			done >> %s/%s
		done """ % (tempdir,subject_list, tempdir,surface,surface,tempdir,cmd_reg) )

	if opts.parallel:
		os.system("cat %s/%s | parallel -j %d" % (tempdir,cmd_reg, numcore) )
	else:
		os.system("while read -r i; do eval $i; done < %s/%s" % (tempdir,cmd_reg) )

	os.chdir('%s/' % tempdir)
	print "Performing smoothing with FWHM = 3.0mm "
	os.system("""
		for hemi in lh rh; do
			for i in ${hemi}*.mgh; do
				temp_outname=$(basename ${i} .00.mgh).03B.mgh
				echo $FREESURFER_HOME/bin/mri_surf2surf --hemi ${hemi} --s fsaverage --sval ${i} --tval ${temp_outname} --fwhm-trg 3 --noreshape --cortex 
			done >> %s
		done""" % (cmd_smooth) )

	if opts.parallel:
		os.system("cat %s | parallel -j %d;" % (cmd_smooth, numcore))
	else:
		os.system("while read -r i; do eval $i; done < %s" % (cmd_smooth) )
	os.chdir('../')
	print "Merging surface images"
	# painful solution to join strings to maintain subject order
	lh_00 = " ".join([("%s/lh." % tempdir) + s + '*.00.mgh' for s in subjects])
	lh_03 = " ".join([("%s/lh." % tempdir) + s + '*.03B.mgh' for s in subjects])
	rh_00 = " ".join([("%s/rh." % tempdir) + s + '*.00.mgh' for s in subjects])
	rh_03 = " ".join([("%s/rh." % tempdir) + s + '*.03B.mgh' for s in subjects])

	os.system("""
			tm_tools merge-images --vertex -o lh.all.%s.00.mgh -i %s
			tm_tools merge-images --vertex -o lh.all.%s.03B.mgh -i %s
			tm_tools merge-images --vertex -o rh.all.%s.00.mgh -i %s
			tm_tools merge-images --vertex -o rh.all.%s.03B.mgh -i %s
		""" % (surface,lh_00,surface,lh_03,surface,rh_00,surface,rh_03))

	if opts.fwhm:
		for i in xrange(len(opts.fwhm)):
			os.system("""
			$FREESURFER_HOME/bin/mri_surf2surf --hemi lh --s fsaverage --sval lh.all.%s.00.mgh --fwhm %d --cortex --tval lh.all.%s.0%dB.mgh
			$FREESURFER_HOME/bin/mri_surf2surf --hemi rh --s fsaverage --sval rh.all.%s.00.mgh --fwhm %d --cortex --tval rh.all.%s.0%dB.mgh
			""" % (surface,int(opts.fwhm[i]),surface,int(opts.fwhm[i]),surface,int(opts.fwhm[i]),surface,int(opts.fwhm[i])))

	# Clean-up
	if opts.noclean:
		exit()
	else:
		os.system("rm -rf %s" % tempdir)


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
