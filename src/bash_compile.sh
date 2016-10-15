#! /bin/bash

while getopts "emh" opt; do
	case $opt in
		e)
			echo "export TFCE_mediation="`dirname $PWD` >> ~/.bashrc
		;;
		m)
			cp ../adjacency_sets/?h.midthickness $SUBJECTS_DIR/fsaverage/surf/
		;;
		h)
			echo "Usage: `basename $0`"
			echo "Options:"
			echo "[-e] Set TFCE_mediation as environment variable in ~/.bashrc"
			echo "[-m] Copy midthickness surface to fsaverage"
			exit 1;
		;;
		\?)
		echo "Invalid option: -$OPTARG"
		exit 1
		;;
	esac
done

mkdir -p ../cython
mkdir -p ../tools/cython

#numstats
python setup.py build_ext --inplace
mv cy_numstats.c ../cython
mv cy_numstats.so ../cython

#tfce
mv TFCE.cpp ../cython
mv TFCE.so ../cython
cp TFCE_.hxx ../cython

#adjacency
mv Adjacency.so ../vertex_tools/cython
mv Adjacency.cpp ../vertex_tools/cython

