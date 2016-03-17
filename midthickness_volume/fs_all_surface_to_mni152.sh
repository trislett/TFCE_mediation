#! /bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: $0 [surface (area or thickness)]"
	exit 1;
fi

surface_type=$1

mri_surf2vol --surfval lh.all.${surface_type}.03B.mgh --hemi lh --outvol lh.all.${surface_type}.03B.mni152.nii.gz --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat

mri_surf2vol --surfval rh.all.${surface_type}.03B.mgh --hemi rh --outvol rh.all.${surface_type}.03B.mni152.nii.gz --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat




