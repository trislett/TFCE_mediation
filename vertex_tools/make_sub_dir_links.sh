#! /bin/bash

analysis_name=$1
surface=$2
dirlocation=`pwd`

if [ $# -eq 0 ]; then
	echo "Usage: $0 [analysis_name] [surface (area or thickness)]"
	echo "Makes a subdirectory for analysis."
	exit 1;
fi

mkdir ${analysis_name}
cd ${analysis_name}

ln -s ${dirlocation}/lh.all.${surface}.03B.mgh lh.all.${surface}.03B.mgh
ln -s ${dirlocation}/rh.all.${surface}.03B.mgh rh.all.${surface}.03B.mgh
#ln -s ${dirlocation}/mask.${surface}.03B.mni152.nii.gz mask.${surface}.03B.mni152.nii.gz

