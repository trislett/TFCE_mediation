#! /bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: $0 [1-p FWE correct volume]"
	exit 1;
fi

mkdir -p cluster_results

result_vol=$1
thresh=$2

if [ "$2" == "" ]; then
	thresh=0.95 # default value
fi

cluster -i ${result_vol} -t ${thresh} --mm --scalarname="1-p" -o cluster_results/$(basename $result_vol .nii.gz)_clusters > cluster_results/$(basename $result_vol .nii.gz)_results



