#! /bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: `basename $0` [-l subject_list] [-s area or thickness]"
	echo "Options:"
	echo "[-p] use GNU parallel"
	echo "[-m] create mean surface images"
	echo "[-f FWHM] specify FWHM smoothing (other than 0,3,and 10). E.g., -f 2"
	echo "[-v] create midthickness MNI152 mean and mask volumes"
	exit 1;
fi

while getopts "s:l:vmf:p" opt; do
	case $opt in
		s)
			surface=$OPTARG
		;;
		l)
			subject_file=$OPTARG
		;;
		v)
			surf_opt=1
		;;
		m)
			means_opt=1
		;;
		f)
			other_FWHM=1
			FWHM=$OPTARG
		;;
		p)
			use_parallel=1
		;;
		\?)
		echo "Invalid option: -$OPTARG"
		exit 1
		;;
	esac
done

echo "Current SUBJECTS_DIR is: " $SUBJECTS_DIR;

for i in $(cat ${subject_file}); do echo -ne $(echo --s $i)" "; done > longname

if [[ $use_parallel = 1 ]]; then
	echo "using parallel processing"
	for i in lh rh; do 
		echo 'mris_preproc' $(cat longname) '--target fsaverage --hemi '$(echo ${i})' --meas '$(echo ${surface})' --out '$(echo ${i})'.all.'$(echo ${surface})'.00.mgh'
	done > cmd_step1
	cat cmd_step1 | parallel -j 2
	rm cmd_step1
else
	eval $(echo 'mris_preproc' $(cat longname) '--target fsaverage --hemi lh --meas ${surface} --out lh.all.${surface}.00.mgh')
	eval $(echo 'mris_preproc' $(cat longname) '--target fsaverage --hemi rh --meas ${surface} --out rh.all.${surface}.00.mgh')
fi
rm longname

if [[ $use_parallel = 1 ]]; then
	for j in lh rh; do 
		echo mri_surf2surf --hemi ${j} --s fsaverage --sval ${j}.all.${surface}.00.mgh --fwhm 10 --cortex --tval ${j}.all.${surface}.10B.mgh
		echo mri_surf2surf --hemi ${j} --s fsaverage --sval ${j}.all.${surface}.00.mgh --fwhm 3 --cortex --tval ${j}.all.${surface}.03B.mgh
	done > cmd_step2
	cat cmd_step2 | parallel -j 4
	rm cmd_step2
else
	mri_surf2surf --hemi lh --s fsaverage --sval lh.all.${surface}.00.mgh --fwhm 10 --cortex --tval lh.all.${surface}.10B.mgh
	mri_surf2surf --hemi rh --s fsaverage --sval rh.all.${surface}.00.mgh --fwhm 10 --cortex --tval rh.all.${surface}.10B.mgh

	mri_surf2surf --hemi lh --s fsaverage --sval lh.all.${surface}.00.mgh --fwhm 3 --cortex --tval lh.all.${surface}.03B.mgh
	mri_surf2surf --hemi rh --s fsaverage --sval rh.all.${surface}.00.mgh --fwhm 3 --cortex --tval rh.all.${surface}.03B.mgh
fi


if [[ $means_opt = 1 ]]; then
	mri_concat lh.all.${surface}.00.mgh --o lh.mean.${surface}.00.mgh --mean
	mri_concat rh.all.${surface}.00.mgh --o rh.mean.${surface}.00.mgh --mean

	mri_concat lh.all.${surface}.03B.mgh --o lh.mean.${surface}.03B.mgh --mean
	mri_concat rh.all.${surface}.03B.mgh --o rh.mean.${surface}.03B.mgh --mean
fi

if [[ $other_FWHM = 1 ]]; then
	mri_surf2surf --hemi lh --s fsaverage --sval lh.all.${surface}.00.mgh --fwhm $FWHM --cortex --tval lh.all.${surface}.0${FWHM}B.mgh
	mri_surf2surf --hemi rh --s fsaverage --sval rh.all.${surface}.00.mgh --fwhm $FWHM --cortex --tval rh.all.${surface}.0${FWHM}B.mgh
fi

if [[ $surf_opt = 1 ]]; then
	mri_surf2vol --surfval lh.mean.${surface}.03B.mgh --hemi lh --outvol lh.mean.${surface}.03B.mni152.nii.gz --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat
	mri_surf2vol --surfval rh.mean.${surface}.03B.mgh --hemi rh --outvol rh.mean.${surface}.03B.mni152.nii.gz --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat
	fslmaths lh.mean.${surface}.03B.mni152.nii.gz -add rh.mean.${surface}.03B.mni152.nii.gz mean.${surface}.03B.mni152.nii.gz
	fslmaths mean.${surface}.03B.mni152.nii.gz -bin mask.${surface}.03B.mni152
	echo "Mean St_Dev Min Max Voxels"
	fslstats mean.${surface}.03B.mni152.nii.gz -M -S -P 0 -P 100 -V
fi
