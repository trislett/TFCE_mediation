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

DESCRIPTION = "Python pipe to build Freesurfer template surface using mris_preproc. $SUBJECT_DIR must be declared (e.g., export SUBJECT_DIR=/path/to/freesurfer/subjects)"

def getArgumentParser(parser = argparse.ArgumentParser(description = DESCRIPTION)):
	parser.add_argument("-i", "--input", 
		nargs=2, 
		help="[subject_list] [surface]", 
		metavar=('*.csv', 'area or thickness'),
		required=True)
	parser.add_argument("-f", "--fwhm", 
		nargs='?', 
		help="Optional. Specify FWHM smoothing (other than 0,3,and 10). E.g., -f 2]", 
		metavar=('INT'))
	parser.add_argument("-p", "--parallel", 
		help="Optional. Use GNU parallel processing", 
		action="store_true")
	return parser

def run(opts):
	subjects = np.genfromtxt(str(opts.input[0]), delimiter=",", dtype=None)
	surface = str(opts.input[1])

	os.system('echo "Current SUBJECTS_DIR is: " $SUBJECTS_DIR;')
	longname='{1}{0}'.format(' --s '.join(subjects), '--s ')
	with open("longname", "w") as text_file:
		text_file.write("%s" % longname)

	if opts.parallel:
		## create template
		os.system(""" 
			echo "using parallel processing"
			for i in lh rh; do 
				echo '$FREESURFER_HOME/bin/mris_preproc' $(cat longname) '--target fsaverage --hemi '$(echo ${i})' --meas '$(echo %s)' --out '$(echo ${i})'.all.'$(echo %s)'.00.mgh';
			done > cmd_step1;
			cat cmd_step1 | parallel -j 2;
			rm cmd_step1;
			rm longname;
			""" % (surface,surface))
		## apply FWHM of 3mm and 10m to template
		os.system("""
			for j in lh rh; do 
					echo $FREESURFER_HOME/bin/mri_surf2surf --hemi ${j} --s fsaverage --sval ${j}.all.%s.00.mgh --fwhm 10 --cortex --tval ${j}.all.%s.10B.mgh
					echo $FREESURFER_HOME/bin/mri_surf2surf --hemi ${j} --s fsaverage --sval ${j}.all.%s.00.mgh --fwhm 3 --cortex --tval ${j}.all.%s.03B.mgh
			done > cmd_step2;
			cat cmd_step2 | parallel -j 4;
			rm cmd_step2;
				""" % (surface,surface,surface,surface))
	else:
		os.system("""
			eval $(echo '$FREESURFER_HOME/bin/mris_preproc' $(cat longname) '--target fsaverage --hemi lh --meas %s --out lh.all.%s.00.mgh');
			eval $(echo '$FREESURFER_HOME/bin/mris_preproc' $(cat longname) '--target fsaverage --hemi rh --meas %s --out rh.all.%s.00.mgh');
			rm longname;
			""" % (surface,surface,surface,surface))
		os.system("""
			$FREESURFER_HOME/bin/mri_surf2surf --hemi lh --s fsaverage --sval lh.all.%s.00.mgh --fwhm 10 --cortex --tval lh.all.%s.10B.mgh
			$FREESURFER_HOME/bin/mri_surf2surf --hemi rh --s fsaverage --sval rh.all.%s.00.mgh --fwhm 10 --cortex --tval rh.all.%s.10B.mgh
			$FREESURFER_HOME/bin/mri_surf2surf --hemi lh --s fsaverage --sval lh.all.%s.00.mgh --fwhm 3 --cortex --tval lh.all.%s.03B.mgh
			$FREESURFER_HOME/bin/mri_surf2surf --hemi rh --s fsaverage --sval rh.all.%s.00.mgh --fwhm 3 --cortex --tval rh.all.%s.03B.mgh
				""" % (surface,surface,surface,surface,surface,surface,surface,surface))
	if opts.fwhm:
		for i in xrange(len(opts.fwhm)):
			os.system("""
			$FREESURFER_HOME/bin/mri_surf2surf --hemi lh --s fsaverage --sval lh.all.%s.00.mgh --fwhm %d --cortex --tval lh.all.%s.%dB.mgh
			$FREESURFER_HOME/bin/mri_surf2surf --hemi rh --s fsaverage --sval rh.all.%s.00.mgh --fwhm %d --cortex --tval rh.all.%s.%dB.mgh
			""" % (surface,int(opts.fwhm[i]),surface,int(opts.fwhm[i]),surface,int(opts.fwhm[i]),surface,int(opts.fwhm[i])))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
