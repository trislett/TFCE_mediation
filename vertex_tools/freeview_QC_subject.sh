#! /bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: `basename $0` [subject]"
	exit 1;
fi

SubjID=$1
dirlocation=`pwd`

freeview -v $SUBJECTS_DIR/$SubjID/mri/T1.mgz $SUBJECTS_DIR/$SubjID/mri/wm.mgz $SUBJECTS_DIR/$SubjID/mri/brainmask.mgz $SUBJECTS_DIR/$SubjID/mri/aseg.mgz:colormap=lut:opacity=0.2 -f $SUBJECTS_DIR/$SubjID/surf/lh.white:edgecolor=blue $SUBJECTS_DIR/$SubjID/surf/lh.pial:edgecolor=red $SUBJECTS_DIR/$SubjID/surf/rh.white:edgecolor=blue $SUBJECTS_DIR/$SubjID/surf/rh.pial:edgecolor=red
