#! /bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: $0 [cluster_annot] [cluster_num] [all_surface_file]"
	echo "e.g. $0 cluster_results/tstat_rh_surf_area_con1_TFCE_FWEcorrP_clusters.nii.gz 3 ../rh.all.area.03B.mgh"
	exit 1;
fi

cd cluster_results

cluster_annot=$(basename $1)
cluster_num=$2
all_surface=$3
base_surface=$(basename $all_surface)
hemi=${base_surface:0:2}

fslmaths ${cluster_annot} -thr ${cluster_num} -uthr ${cluster_num} -bin $(basename $cluster_annot _clusters.nii.gz)_cluster${cluster_num}_mask

if [ ! -f $(basename $all_surface .mgh).mni152.nii.gz ]; then
	mri_surf2vol --surfval ../${all_surface} --hemi ${hemi} --outvol $(basename $all_surface .mgh).mni152.nii.gz --projfrac 0.5 --template $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --reg $FREESURFER_HOME/average/mni152.register.dat
fi

fslmaths $(basename $all_surface .mgh).mni152.nii.gz -mul $(basename $cluster_annot _clusters.nii.gz)_cluster${cluster_num}_mask.nii.gz $(basename $all_surface .mgh)_cluster${cluster_num}

fslstats -t $(basename $all_surface .mgh)_cluster${cluster_num} -M > $(basename $cluster_annot _clusters.nii.gz)_mean_cluster${cluster_num}

rm $(basename $all_surface .mgh)_cluster${cluster_num}.nii.gz
