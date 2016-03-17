#! /bin/bash

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
mv Adjacency.so ../tools/cython
mv Adjacency.cpp ../tools/cython

