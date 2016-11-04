#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm, inv
import argparse as ap

def sym(w):
	return w.dot(inv(sqrtm(w.T.dot(w))))
def writeCSV(base,tail,data):
	np.savetxt(("%s_%s.csv" % (base,tail)), data, delimiter=",", fmt='%10.8f')

DESCRIPTION = """
Simple program to condition the regressors for TFCE_mediation analyses. The program returns either the orthogonalization (i.e., --orthogonalize) of the input file(s) or it returns the residuals (ie. --residuals) from a least squares regression to remove the effect of covariates from variable.

Orthogonalization or the residuals should be used if the predictor variables and covariates are not completely independent from each other. Orthogonalization should not be performed for mediation analysis.
"""

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION, formatter_class=ap.RawTextHelpFormatter)):

	#input type
	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument("-i", "--input",
		help="Two csv files: [regressor(s)] [covariates]. Please note that if the -r option is used, the regressor(s) will be treated as dependent variable(s).", 
		nargs=2, 
		metavar=('*.csv', '*.csv'))
	group.add_argument("-f", "--file",  
		help="One csv file: [regressors(s)]", 
		nargs=1, 
		metavar=('*.csv'))

	#options
	ap.add_argument("-d", "--demean", 
		help="demean columns", 
		action="store_true")
	ap.add_argument("-s", "--stddemean", 
		help="demean and standardize columns", 
		action="store_true")

	#which tool to use
	proceducetype = ap.add_mutually_exclusive_group(required=True)
	proceducetype.add_argument("-o", "--orthogonalize", 
		help="orthogonalize the inputs", 
		action="store_true")
	proceducetype.add_argument("-r", "--residuals", 
		help="residuals after regressing covariates", 
		action="store_true")
	proceducetype.add_argument("-j", "--juststandarize", 
		help="Just demean or standardize the regressors. i.e., no regression or orthogonization",
		action="store_true")

	return ap


def run(opts):
	if opts.orthogonalize:
		if opts.file:
			regressors = np.genfromtxt(opts.file[0], delimiter=',')
			regressors_nocsv = opts.file[0].split('.csv',1)[0]
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
			writeCSV(regressors_nocsv,'orthogonal',out_regressors)
		if opts.input:
			if covars.ndim==1:
				out_pred = out_regressors[:,0]
				out_covars = regressors_orthog[:,-1:]
			else:
				out_pred = out_regressors[:,:-(covars.shape[1])]
				out_covars = regressors_orthog[:,-covars.shape[1]:]
			writeCSV(pred_nocsv,'orthogonal',out_pred)
			writeCSV(covars_nocsv,'orthogonal',out_covars)

	elif opts.residuals:
		# check number of files
		if opts.file:
			print "Two *.csv files are necesssary"
			exit()
		arg_depvars = str(opts.input[0])
		arg_covars = str(opts.input[1])
		depvars_nocsv = arg_depvars.split('.csv',1)[0]
		covars_nocsv = arg_covars.split('.csv',1)[0]
		depvars = np.genfromtxt(arg_depvars, delimiter=',')
		covars = np.genfromtxt(arg_covars, delimiter=',')
		# check number of columns for dependent variable
		if depvars.ndim>1:
			print "The dependent variable should only only have one column"
			exit()
		if opts.demean:
			covars = covars - np.mean(covars, axis=0)
			writeCSV(covars_nocsv,'dm',covars)
		if opts.stddemean:
			covars = covars - np.mean(covars, axis=0)
			covars = np.divide(covars,np.std(covars,axis=0))
			writeCSV(covars_nocsv,'std_dm',covars)
		x_covars = np.column_stack([np.ones(covars.shape[0]),covars])
		a_c = np.linalg.lstsq(x_covars, depvars)[0]
		resids = depvars - np.dot(x_covars,a_c)
		writeCSV(depvars_nocsv,'resids',resids)

	elif opts.juststandarize:
		if opts.file:
			regressors = np.genfromtxt(opts.file[0], delimiter=',')
			regressors_nocsv = opts.file[0].split('.csv',1)[0]
			if opts.demean:
				regressors = regressors - np.mean(regressors, axis=0)
				writeCSV(regressors_nocsv,'dm',regressors)
			elif opts.stddemean:
				regressors = regressors - np.mean(regressors, axis=0)
				regressors = np.divide(regressors,np.std(regressors,axis=0))
				writeCSV(regressors_nocsv,'std_dm',regressors)
			else:
				print "Please select demean (-d) or standardize (-s)"
				exit()
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
			elif opts.stddemean:
				covars = covars - np.mean(covars, axis=0)
				covars = np.divide(covars,np.std(covars,axis=0))
				pred = pred - np.mean(pred, axis=0)
				pred = np.divide(pred,np.std(pred,axis=0))
				writeCSV(covars_nocsv,'std_dm',covars)
				writeCSV(pred_nocsv,'std_dm',pred)
			else:
				print "Please select demean (-d) or standardize (-s)"
				exit()

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
