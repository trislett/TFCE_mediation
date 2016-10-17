#! /bin/bash


if [ $# -eq 0 ]; then
	echo "Create TBSS skeleton. To be used after ./nl_FAtoStd.sh has been completed for all subjects."
	echo "Usage: `basename $0` -t <threshold> [-options]"
	echo "-t [threshold] (e.g. -t 0.2)"
	echo "Options:"
	echo "[-d] Speficy FAtoStd directory (the full directory must be specified)" 
	echo "[-m] Multiply skeleton mask by the T1toStdMask for each subject"
	echo "[-l] use a list of subjects"
	echo "[-s] use ANTS skeletonization"
	echo "[-p INT] use GNU parallel with specifying number of cores (e.g. -p 8)"
	exit 1;
fi

SCRIPT=$0
SCRIPTPATH=`dirname $SCRIPT`
FA2SDDIR=../FAtoStd
thr=0.2

## options
while getopts "t:d:ml:sp:" opt; do
	case $opt in
		t)
			thr=$OPTARG
		;;
		d)
			FA2SDDIR=$OPTARG
		;;
		m)
			mask_opt=1
		;;
		l)
			list_opt=1
			subject_file=$OPTARG
		;;
		s)
			ants_opt=1
		;;
		p)
			use_parallel=1
			numcore=$OPTARG
		;;
		\?)
		echo "Invalid option: -$OPTARG"
		exit 1
		;;
	esac
done

mkdir -p stats && cd stats

echo "creating average mask"
if [[ $list_opt = 1 ]]; then
	for i in $(cat ../${subject_file}); do echo -ne ${FA2SDDIR}/$(echo $i)*FAtoStd*" "; done > listFAtoStd
	for i in $(cat ../${subject_file}); do echo -ne ${FA2SDDIR}/$(echo $i)*_braintoStd_mask.nii.gz" "; done > brainT1toStdMask
	eval $(echo 'AverageImages 3 averageFA.nii.gz 0 ' $(cat listFAtoStd))
	fslmaths averageFA.nii.gz -thr 0.01 -bin averageFA_mask
	if [[ $mask_opt = 1 ]]; then
		echo 'correcting mask for all subjects'
		cp averageFA_mask.nii.gz averageFA_mask_original.nii.gz
		cp averageFA.nii.gz averageFA_original.nii.gz
		eval $(echo '${SCRIPTPATH}/createFinalMask.py -i averageFA_mask.nii.gz -m ' $(cat brainT1toStdMask))
	fi
	fslmaths averageFA.nii.gz -mul averageFA_mask.nii.gz averageFA.nii.gz
	rm listFAtoStd
	rm brainT1toStdMask
else
	AverageImages 3 averageFA.nii.gz 0 ${FA2SDDIR}/*FAtoStd*
	fslmaths averageFA.nii.gz -thr 0.01 -bin averageFA_mask
	if [[ $mask_opt = 1 ]]; then
		echo 'correcting mask for all subjects'
		cp averageFA_mask.nii.gz averageFA_mask_original.nii.gz
		cp averageFA.nii.gz averageFA_original.nii.gz
		${SCRIPTPATH}/createFinalMask.py -i averageFA_mask.nii.gz -m ${FA2SDDIR}/*_braintoStd_mask.nii.gz
	fi
	fslmaths averageFA.nii.gz -mul averageFA_mask.nii.gz averageFA.nii.gz
fi

echo "creating skeleton mask"
if [[ $ants_opt = 1 ]]; then
	ThresholdImage 3 averageFA.nii.gz averageFASeg.nii.gz ${thr} Inf
	ImageMath 3 averageFASeg.nii.gz GetLargestComponent averageFASeg.nii.gz
	skel.sh averageFASeg.nii.gz averageFASeg 1
	fslmaths averageFASeg_topo_skel.nii.gz -bin mean_FA_skeleton_mask
	fslmaths averageFA_mask.nii.gz -mul -1 -add 1 -add mean_FA_skeleton_mask.nii.gz mean_FA_skeleton_mask_dst
else
	tbss_skeleton -i averageFA.nii.gz -o averageFA_Skel
	fslmaths averageFA_Skel.nii.gz -thr ${thr} -bin mean_FA_skeleton_mask
	fslmaths averageFA_mask.nii.gz -mul -1 -add 1 -add mean_FA_skeleton_mask.nii.gz mean_FA_skeleton_mask_dst
fi
distancemap -i mean_FA_skeleton_mask_dst -o mean_FA_skeleton_mask_dst

echo "applying skeleton to subjects"
if [[ $list_opt = 1 ]]; then
	for i in $(cat ../${subject_file}); do echo tbss_skeleton -i averageFA.nii.gz -p ${thr} mean_FA_skeleton_mask_dst.nii.gz ${FSLDIR}/data/standard/LowerCingulum_1mm ${FA2SDDIR}/$(echo $i)*FAtoStd* skel_$(echo $i)_FA -s mean_FA_skeleton_mask.nii.gz; done > cmd_skel_parallel
else
	for i in ${FA2SDDIR}/*FAtoStd*; do echo tbss_skeleton -i averageFA.nii.gz -p ${thr} mean_FA_skeleton_mask_dst.nii.gz ${FSLDIR}/data/standard/LowerCingulum_1mm $i skel_$(basename $i .nii.gz) -s mean_FA_skeleton_mask.nii.gz; done > cmd_skel_parallel
fi

if [[ $use_parallel = 1 ]]; then
	cat cmd_skel_parallel | parallel -j ${numcore}
else
	while read -r i; do eval $i; done <cmd_skel_parallel
fi

${SCRIPTPATH}/mergeNifTi.py -o all_FA_skeletonised.nii.gz -m mean_FA_skeleton_mask.nii.gz -i skel_*
rm skel_*
rm cmd_skel_parallel
