#!/usr/bin/python

import os
import sys
import numpy as np
from scipy.linalg import sqrtm, inv
import argparse as ap

def sym(w):
	return w.dot(inv(sqrtm(w.T.dot(w))))

ap = ap.ArgumentParser(description=""" 	Simple program to demean and orthogonalization csv file(s) for cortex-wise
										multiple regression and mediation analyses.
										e.g. %s -i pred.csv covars.csv -s""" % (sys.argv[0]))

parser = ap.add_argument("-d", "--demean", help="demean columns prior to orthogonalization", action="store_true")
parser = ap.add_argument("-s", "--stddemean", help="demean and standardize columns prior to orthogonalization", action="store_true")
group = ap.add_mutually_exclusive_group(required=True)
group.add_argument("-r", "--csv",  help="Single csv: [regressors] file", nargs=1, metavar=('*.csv'))
group.add_argument("-i", "--input",help="Two csv: [dependent_var] [covariates]", nargs=2, metavar=('*.csv', '*.csv'))

opts = ap.parse_args()

if opts.csv:
	regressors = np.genfromtxt(opts.csv[0], delimiter=',')
	regressors_nocsv = opts.csv[0].split('.csv',1)[0]

	if opts.demean:
		regressors = regressors - np.mean(regressors, axis=0)
	if opts.stddemean:
		regressors = regressors - np.mean(regressors, axis=0)
		regressors = np.divide(regressors,np.std(regressors,axis=0))

	n = regressors.shape[0]
	regressors_ones = np.column_stack([np.ones(n),regressors])
	regressors_orthog = sym(regressors_ones)
	out_regressors = regressors_orthog[:,1:] 
	np.savetxt("%s_orthogonal.csv" % (regressors_nocsv), out_regressors, delimiter=",")

if opts.input:
	pred = np.genfromtxt(opts.input[0], delimiter=',')
	covars = np.genfromtxt(opts.input[1], delimiter=',')
	pred_nocsv = opts.input[0].split('.csv',1)[0]
	covars_nocsv = opts.input[1].split('.csv',1)[0]
	regressors = np.column_stack([pred,covars])

	if opts.demean:
		regressors = regressors - np.mean(regressors, axis=0)
	if opts.stddemean:
		regressors = regressors - np.mean(regressors, axis=0)
		regressors = np.divide(regressors,np.std(regressors,axis=0))

	n = regressors.shape[0]
	regressors_ones = np.column_stack([np.ones(n),regressors])
	pred_ones = np.column_stack([np.ones(n),pred]) # fix later
	regressors_orthog = sym(regressors_ones)
	out_regressors = regressors_orthog[:,1:]
	out_pred = out_regressors[:,:-(covars.shape[1])]
	out_covars = regressors_orthog[:,pred_ones.shape[1]:]
	np.savetxt("%s_orthogonal.csv" % (pred_nocsv), out_pred, delimiter=",")
	np.savetxt("%s_orthogonal.csv" % (covars_nocsv), out_covars, delimiter=",")
