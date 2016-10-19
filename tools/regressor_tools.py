#!/usr/bin/python

import sys
import numpy as np
import statsmodels.api as sm
from scipy.linalg import sqrtm, inv
import argparse as ap

def sym(w):
	return w.dot(inv(sqrtm(w.T.dot(w))))
def writeCSV(base,tail,data):
	np.savetxt(("%s_%s.csv" % (base,tail)), data, delimiter=",", fmt='%10.5f')

ap = ap.ArgumentParser(description="""
Simple program to condition the regressors for TFCE_mediation analyses. 
Either orthogonalization or removing the effect confounders from a regressor 
variable. Orthogonalization should not be performed for mediation analysis.

e.g. %s -o -i pred.csv covars.csv -s""" % (sys.argv[0]))

proceducetype = ap.add_mutually_exclusive_group(required=True)
proceducetype.add_argument("-o", "--orthogonalize", help="orthogonalize the inputs")
proceducetype.add_argument("-r", "--residuals", help="regress the dependent variable by the covariates, and store the residual")
proceducetype.add_argument("-j", "--juststandarize", help="Just demean or standardize the regressors. i.e., no regression or orthogonization")

#options
parser = ap.add_argument("-d", "--demean", help="demean columns", action="store_true")
parser = ap.add_argument("-s", "--stddemean", help="demean and standardize columns", action="store_true")

#input type
group = ap.add_mutually_exclusive_group(required=True)
group.add_argument("-i", "--input",help="Two csv: [dependent_var] [covariates]", nargs=2, metavar=('*.csv', '*.csv'))
group.add_argument("-f", "--file",  help="One csv: [regressors] file", nargs=1, metavar=('*.csv'))

opts = ap.parse_args()

if opts.orthogonalize:
	if opts.file:
		regressors = np.genfromtxt(opts.csv[0], delimiter=',')
		regressors_nocsv = opts.csv[0].split('.csv',1)[0]
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
	regressors_orthog = sym(regressors_ones)
	out_regressors = regressors_orthog[:,1:]

	if opts.file:
		writeCSV(regressors_nocsv,'orthogonal',out_regressors):
	if opts.input:
		out_pred = out_regressors[:,:-(covars.shape[1])]
		out_covars = regressors_orthog[:,pred_ones.shape[1]:]
		writeCSV(pred_nocsv,'orthogonal',out_pred)
		writeCSV(covars_nocsv,'orthogonal',out_covars)

elif opts.residuals:
	if opts.file:
		print "Two *.csv files are necesssary"
		exit()
	arg_depvars = str(opts.input[0])
	arg_covars = str(opts.input[1])
	depvars_nocsv = arg_depvars.split('.csv',1)[0]
	covars_nocsv = arg_covars.split('.csv',1)[0]
	depvars = np.genfromtxt(arg_depvars, delimiter=',')
	covars = np.genfromtxt(arg_covars, delimiter=',')
	if opts.demean:
		covars = covars - np.mean(covars, axis=0)
		writeCSV(covars_nocsv,'dm',covars)
	if opts.stddemean:
		covars = covars - np.mean(covars, axis=0)
		covars = np.divide(covars,np.std(covars,axis=0))
		writeCSV(covars_nocsv,'std_dm',covars)
	x_covars = sm.add_constant(covars)
	a_c = np.linalg.lstsq(x_covars, depvars)[0]
	resids = depvars - np.dot(x_covars,a_c)
	writeCSV(depvars_nocsv,'resids',resids)

elif opts.juststandarize:
	if opts.file:
		regressors = np.genfromtxt(opts.csv[0], delimiter=',')
		regressors_nocsv = opts.csv[0].split('.csv',1)[0]
		if opts.demean:
			regressors = regressors - np.mean(regressors, axis=0)
			writeCSV(regressors_nocsv,'dm',regressors)
		if opts.stddemean:
			regressors = regressors - np.mean(regressors, axis=0)
			regressors = np.divide(regressors,np.std(regressors,axis=0))
			writeCSV(regressors_nocsv,'std_dm',regressors)
	if opts.input:
		pred = np.genfromtxt(opts.input[0], delimiter=',')
		covars = np.genfromtxt(opts.input[1], delimiter=',')
		pred_nocsv = opts.input[0].split('.csv',1)[0]
		covars_nocsv = opts.input[1].split('.csv',1)[0]
		if opts.demean:
			covars = covars - np.mean(covars, axis=0)
			pred = pred - np.mean(pred, axis=0)
			writeCSV(covars_nocsv,'dm',covars)
			writeCSV(pred_nocsv,'dm',pred)
		if opts.stddemean:
			covars = covars - np.mean(covars, axis=0)
			covars = np.divide(covars,np.std(covars,axis=0))
			pred = pred - np.mean(pred, axis=0)
			pred = np.divide(covars,np.std(pred,axis=0))
			writeCSV(covars_nocsv,'std_dm',covars)
			writeCSV(pred_nocsv,'std_dm',pred)
