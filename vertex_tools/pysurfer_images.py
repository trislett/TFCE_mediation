#!/usr/bin/python

import sys
import os
from surfer import Brain

os.system("export ETS_TOOLKIT=qt4; export QT_API=pyqt")

#img_data_lh = nib.freesurfer.mghformat.load(lh_surf))
#data_full_lh = img_data_lh.get_data()
#data_lh = np.squeeze(data_full_lh)
#img_data_rh = nib.freesurfer.mghformat.load(lh_surf))
#data_full_rh = img_data_rh.get_data()
#data_rh = np.squeeze(data_full_rh)

if len(sys.argv) < 5:
	print "Usage: %s [analysis_name] [fsaverage_surf] [overlay_lh] [overlay_rh]" % (str(sys.argv[0]))
else:
	cmdargs = str(sys.argv)
	analysis_name = str(sys.argv[1])
	fsaverage_surf = str(sys.argv[2])
	overlay_lh = str(sys.argv[3])
	overlay_rh = str(sys.argv[4])

	subject_id = 'fsaverage'
	hemi = 'split'
	surface = fsaverage_surf

	brain = Brain(subject_id, hemi, surface, views=['lat', 'med'])
	brain.add_overlay(overlay_lh, min=.95, max=1, sign="pos",  hemi='lh')
	brain.add_overlay(overlay_rh, min=.95, max=1, sign="pos",  hemi='rh')
	brain.save_image('%s_hemi.png' % analysis_name)
	brain.close()

	hemi = 'both'
	brain = Brain(subject_id, hemi, surface, views=['rostral'])
	brain.add_overlay(overlay_lh, min=.95, max=1, sign="pos",  hemi='lh')
	brain.add_overlay(overlay_rh, min=.95, max=1, sign="pos",  hemi='rh')
	brain.save_image('%s_rostral.png' % analysis_name)
	brain.show_view('caudal')
	brain.save_image('%s_caudal.png' % analysis_name)
	brain.show_view('dorsal')
	brain.save_image('%s_dorsal.png' % analysis_name)
	brain.show_view('ventral')
	brain.save_image('%s_ventral.png' % analysis_name)
	brain.close()

#brain.add_data(data_lh,hemi='lh',colormap="heat",min=0.95,max=1)
#brain.add_data(data_lh,hemi='rh',colormap="PuBu",min=0.95,max=1)
