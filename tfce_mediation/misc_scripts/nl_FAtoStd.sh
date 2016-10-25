#! /bin/bash

# Motiation from: Tustison NJ, Avants BB, Cook PA, Kim J, Whyte J, Gee JC, Stone JR. Logical circularity in voxel-based analysis: normalization strategy may induce statistical bias. Hum Brain Mapp. 2014 Mar;35(3):745-59. doi: 10.1002/hbm.22211.

if [ $# -eq 0 ]; then
	echo "Usage: `basename $0` [subX_FA.nii.gz] [subX_T1_brain.nii.gz] [standard_T1.nii.gz]"
	echo "Uses ANTS to non-linear register FA to T1, then T1 to Standard image (e.g. MNI152_T1_1mm_brain.nii.gz). FA images are then warped to standard space without registering FA to FA images."
	exit 1;
fi


FA=$1
subjectT1=$2
stdT1=$3

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

mkdir -p FAtoStd

fslmaths $subjectT1 -bin FAtoStd/$(basename ${subjectT1} .nii.gz)_mask

ANTS 3 -m MI[${subjectT1},${FA},1,32] --use-Histogram-Matching -o FAtoStd/$(basename ${FA} .nii.gz)toT1.nii.gz -i 100x100x100x20

antsApplyTransforms -d 3 -r ${subjectT1} -i ${FA} -e 0 -t FAtoStd/$(basename ${FA} .nii.gz)toT1Warp.nii.gz -t FAtoStd/$(basename ${FA} .nii.gz)toT1Affine.txt -o FAtoStd/$(basename ${FA} .nii.gz)toT1.nii.gz -v 1

fslmaths FAtoStd/$(basename ${FA} .nii.gz)toT1.nii.gz -mul FAtoStd/$(basename ${subjectT1} .nii.gz)_mask FAtoStd/$(basename ${FA} .nii.gz)toT1.nii.gz

antsRegistrationSyNQuick.sh -d 3 -f ${stdT1} -m ${subjectT1} -t s -o FAtoStd/$(basename ${subjectT1} .nii.gz)toStd_ -n 1

antsApplyTransforms -d 3 -r ${stdT1} -i FAtoStd/$(basename ${FA} .nii.gz)toT1.nii.gz -e 0 -t FAtoStd/$(basename ${subjectT1} .nii.gz)toStd_1Warp.nii.gz -t FAtoStd/$(basename ${subjectT1} .nii.gz)toStd_0GenericAffine.mat -o FAtoStd/$(basename ${FA} .nii.gz)toStd.nii.gz -v 1

fslmaths FAtoStd/$(basename ${subjectT1} .nii.gz)toStd_Warped.nii.gz -bin FAtoStd/$(basename ${subjectT1} .nii.gz)toStd_mask

fslmaths FAtoStd/$(basename ${FA} .nii.gz)toStd.nii.gz -mul FAtoStd/$(basename ${subjectT1} .nii.gz)toStd_mask FAtoStd/$(basename ${FA} .nii.gz)toStd.nii.gz
