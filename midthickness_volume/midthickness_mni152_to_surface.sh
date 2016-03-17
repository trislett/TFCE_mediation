#! /bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: $0 [Volume] [hemi (lh or rh)]"
	exit 1;
fi

mni_vol=$1
hemi=$2

mri_vol2surf --hemi ${hemi} --reg $FREESURFER_HOME/average/mni152.register.dat --mov ${mni_vol} --o $(basename $mni_vol .nii.gz).mgh --projfrac 0.5 --trgsubject fsaverage
