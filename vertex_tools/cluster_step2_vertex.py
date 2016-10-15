#!/usr/bin/python

import os
import sys
import numpy as np
import nibabel as nib

if len(sys.argv) < 3:
	print "*******************************************************************"
	print "Usage: %s [4D surface] [label file]" % (str(sys.argv[0]))
	print "  "
	print "Run from cluster results folder"
	print "*******************************************************************"
else:
	cmdargs = str(sys.argv)
	arg_allsurf = str(sys.argv[1])
	label = str(sys.argv[2])
	img_allsurf = nib.freesurfer.mghformat.load(arg_allsurf)
	alldata_full = img_allsurf.get_data()
	os.system("cat %s | tail -n +3 | awk '{ print $1}' > %s.verticeslist" % (label,label))
	vertices = np.genfromtxt("%s.verticeslist" % label, delimiter=',')
	values = np.zeros((len(vertices),alldata_full.shape[3]))
	ite=0
	for i in vertices:
		values[ite,:]=alldata_full[int(i),0,0,:]
 		ite+=1
	meanvalues=np.sum(values,axis=0)/len(vertices)
	np.savetxt(("%s.mean.txt" % label), meanvalues, delimiter=',',fmt='%1.5f')
