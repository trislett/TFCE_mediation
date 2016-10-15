#! /bin/bash

analysis_name=$1
dirlocation=`pwd`

mkdir ${analysis_name}
cd ${analysis_name}
mkdir python_temp
for i in raw_nonzero.npy header_mask.npy affine_mask.npy data_mask.npy num_voxel.npy num_subjects.npy; do
	ln -s ${dirlocation}/python_temp/${i} python_temp/${i}
done

