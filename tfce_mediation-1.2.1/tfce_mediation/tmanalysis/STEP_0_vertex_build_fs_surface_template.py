#!/usr/bin/env python

#    Wrapper for building surface templates for tfce_mediation
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

def merge_surfaces(tempdir,hemi,smoothname,subjects,surface):
	subject_list = " ".join([("%s/%s." % (tempdir,hemi)) + s + ("*.%s.mgh" % smoothname) for s in subjects])
	os.system("tm_tools merge-images --vertex -o %s.all.%s.%s.mgh -i %s" % (hemi,surface,smoothname,subject_list))

def getArgumentParser(parser = argparse.ArgumentParser(description = DESCRIPTION)):
	parser.add_argument("-i", "--input", 
		nargs=2, 
		help="[subject_list] [surface]", 
		metavar=('*.csv', 'area or thickness'),
		required=True)
	parser.add_argument("-f", "--fwhm", 
		nargs='+', 
		help="Optional. Specify FWHM smoothing other than 3mm. Multiple smoothing values can be specificed, E.g.,, for  FWHM = 4mm and 10mm ( -f 4 10 )",
		metavar=('FWHM'))
	parser.add_argument("-p", "--parallel", 
		nargs=1,
		type=int,
		help="Optional (recommended). Use GNU parallel processing with entering the number of cores (e.g., -p 8)", 
		metavar=('INT'))
	parser.add_argument("-g", "--usegeodesicfwhm", 
		nargs=2,
		type=str,
		help="""
		Optional. Use geodesic FHWM smoothing at midthickness surface (no fudge factor). The distances lists must be specifity for the left and right hemispheres, respectively. The default FWHM is 3mm, but other distances can be specified using -f option. Important, this requires creation of distance lists for earh hemisphere using /tfce_mediation/misc_scripts/fwhm_compute_distances_parallel.py or downloading them from the midthickness surface from tm_addons (github.com/trislett/tm_addons). Please note that smoothing for each surface image with geodesicFWHM takes around 5 minutes per subject and uses around 2GB of RAM for the 9mm distance lists. e.g., -g $TM_ADDONS/geodesicFWHM/lh_9.0mm_fwhm_distances.npy $TM_ADDONS/geodesicFWHM/rh_9.0mm_fwhm_distances.npy.
		""", 
		metavar=('STRING'))
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


	if opts.usegeodesicfwhm:
		fwhm = [3.0]
		if opts.fwhm:
			fwhm = opts.fwhm
		for j in fwhm:
			print "Performing geodesic smoothing with FWHM = %smm." % j
			os.system("""
					for i in lh*.mgh; do
						temp_outname=$(basename ${i} .00.mgh).0%dB.mgh
						echo tm_tools geodesic-fwhm --hemi lh -i ${i} -o ${temp_outname} -d %s -f %s
					done >> %s""" % (int(float(j)), opts.usegeodesicfwhm[0],j,cmd_smooth) )
			os.system("""
					for i in rh*.mgh; do
						temp_outname=$(basename ${i} .00.mgh).0%dB.mgh
						echo tm_tools geodesic-fwhm --hemi rh -i ${i} -o ${temp_outname} -d %s -f %s
					done >> %s""" % (int(float(j)),opts.usegeodesicfwhm[1],j,cmd_smooth) )
	else:
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
	merge_surfaces(tempdir,'lh','00',subjects,surface)
	merge_surfaces(tempdir,'rh','00',subjects,surface)

	if opts.usegeodesicfwhm and opts.fwhm:
		for j in fwhm:
			fwhm_name = '0%dB' % int(j)
			merge_surfaces(tempdir,'lh',fwhm_name,subjects,surface)
			merge_surfaces(tempdir,'rh',fwhm_name,subjects,surface)
	else:
		merge_surfaces(tempdir,'lh','03B',subjects,surface)
		merge_surfaces(tempdir,'rh','03B',subjects,surface)

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
