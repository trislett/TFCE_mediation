#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib
import argparse as ap
#from sklearn.decomposition import PCA

ap = ap.ArgumentParser(description=""" Calculate mean of label or annotation values.""")

group = ap.add_mutually_exclusive_group(required=True)
group.add_argument("-l", "--label", metavar=('*_label_surface.mgh'), nargs=1)
group.add_argument("-a", "--annot", metavar=('*annot'), nargs=1)
ap.add_argument("-i", "--input", nargs=1, help="Load 4D surface files", metavar=('?h.all.???.mgh'), required=True)
ap.add_argument("-r", "--range", nargs=2, help="Input range of labels to extract. (e.g.: -r 1 12)", metavar=('INT','INT'))
#ap.add_argument("-p", "--pca", help="[Number of components] Perform PCA on mean FA ROIs.",  type=int, nargs=1, metavar=('INT'))
#ap.add_argument("-m", "--mask", help="Specify mask", nargs=1)

opts = ap.parse_args()

allsurf = opts.input[0]

#load surface data
img_data = nib.freesurfer.mghformat.load(allsurf)
img_data_full = img_data.get_data()
affine_mask = img_data.get_affine()

img_data = np.squeeze(img_data_full)
n = img_data.shape[1]
mean_data = np.sum(img_data,axis=1)/img_data.shape[1]
#outmean_lh[:,0,0] = mean_lh

if opts.label:
	label = opts.label[0]
	lable_name = os.path.basename(opts.label[0].split('.mgh',1)[0])
	img_label = nib.freesurfer.mghformat.load(label)
	data_label = img_label.get_data()
	data_label = np.squeeze(data_label)

	if opts.range:
		lowerLabel = int(opts.range[0])
		upperLabel = int(opts.range[1]) + 1
		num_label = upperLabel - lowerLabel
		print 'Extracting from %d to %d label for a total of %d mean values' % (lowerLabel,upperLabel, num_label)
		outMeanValues = np.zeros([n,num_label])
		counter=0
		for i in range(lowerLabel,upperLabel):
			outMeanValues[:,counter]=img_data[data_label==(i)].mean(axis=0)
			counter+=1
	else:
		num_label = int(data_label.max())
		print 'Extracting %d mean label values' % num_label

		outMeanValues = np.zeros([n,num_label])

		for i in range((num_label)):
			outMeanValues[:,i]=img_data[data_label==(i+1)].mean(axis=0)
	np.savetxt("Mean_%s.csv" % lable_name, outMeanValues, delimiter=",")
elif opts.annot:
	annot_path = opts.annot[0]
	lable_name = os.path.basename(opts.annot[0].split('.annot',1)[0])
	data_label,_,names = nib.freesurfer.io.read_annot(annot_path)
	lowerLabel = 1
	upperLabel = int(len(names))
	num_label = len(names)
	print 'Extracting from %d labels mean values from annotation' % (num_label)
	outMeanValues = np.zeros([n,num_label-1])
	counter=0
	for i in range(lowerLabel,upperLabel):
		outMeanValues[:,counter]=img_data[data_label==(i)].mean(axis=0)
		counter+=1
	np.savetxt("Mean_%s.csv" % lable_name, outMeanValues, delimiter=",")
	np.savetxt("Names_%s.csv" % lable_name,names[1:],fmt='%s',delimiter=",")
#if opts.pca:
#	print("Evaluating %d components" % opts.pca[0])
#	x = (meanvalue - np.mean(meanvalue, 0)) / np.std(meanvalue, 0)
#	pca = PCA(n_components=opts.pca[0])
#	x_r = pca.fit(x).transform(x)
#	np.savetxt(("PCA_components.csv"), x_r, delimiter=",", fmt='%10.5f')
#	np.savetxt(("PCA_var_explained_ratio.csv"), pca.explained_variance_ratio_, delimiter=",", fmt='%10.5f')

