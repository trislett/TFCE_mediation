#!/usr/bin/python

import sys
import numpy as np
import statsmodels.api as sm
import argparse as ap

ap = ap.ArgumentParser(description=""" 	Simple program to output residuals for cortex-wise 											multiple regression and mediation analyses.
										e.g. %s -i dependent.csv covars.csv -s""" % (sys.argv[0]))

ap.add_argument("-i", "--input",help="[dependent_var].csv [covariates].csv", nargs=2, metavar=('*.csv', '*.csv'), required=True)
ap.add_argument("-d", "--demean", help="demean covariates", action="store_true", required=False)
ap.add_argument("-s", "--stddemean", help="demean and standardize covariates", action="store_true", required=False)
ap.add_argument("-o", "--output", help="Specificy output name (default [dependent_var]_resids.csv)", action="store_true", required=False)
opts = ap.parse_args()


arg_depvars = str(opts.input[0])
arg_covars = str(opts.input[1])
depvars_nocsv = arg_depvars.split('.csv',1)[0]
covars_nocsv = arg_covars.split('.csv',1)[0]
depvars = np.genfromtxt(arg_depvars, delimiter=',')
covars = np.genfromtxt(arg_covars, delimiter=',')


if opts.demean:
	covars = covars - np.mean(covars, axis=0)
	np.savetxt(("%s_dm.csv" % covars_nocsv), covars, delimiter=",", fmt='%10.5f')
if opts.stddemean:
	covars = covars - np.mean(covars, axis=0)
	covars = np.divide(covars,np.std(covars,axis=0))
	np.savetxt(("%s_std_dm.csv" % covars_nocsv), covars, delimiter=",", fmt='%10.5f')

x_covars = sm.add_constant(covars)
a_c = np.linalg.lstsq(x_covars, depvars)[0]
resids = depvars - np.dot(x_covars,a_c)

if opts.output:
	np.savetxt(opts.output[0], resids, delimiter=",", fmt='%10.5f')
else:
	np.savetxt(("%s_resids.csv" % depvars_nocsv), resids, delimiter=",", fmt='%10.5f')
