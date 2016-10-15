#! /bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: $0 [1-p FWE correct volume] [hemi] optional: [1-p threshold]"
	echo "Default 1-P threshold is 0.95 (i.e., pFWE<0.05)"
	exit 1;
fi

mkdir -p cluster_results

result_vol=$1
hemi=$2
thresh=$3

if [ "$3" == "" ]; then
	thresh=0.95 # default value
fi

mri_surfcluster --in ${result_vol} --thmin ${thresh} --hemi ${hemi} --subject fsaverage --o cluster_results/$(basename $result_vol .mgh)_maskedvalues.mgh --olab cluster_results/$(basename $result_vol .mgh).label --ocn cluster_results/$(basename $result_vol .mgh)_label_surface.mgh > cluster_results/$(basename $result_vol .mgh).output
