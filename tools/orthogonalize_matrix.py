#!/usr/bin/python

import os
import sys
import numpy as np
import statsmodels.api as sm
from scipy.linalg import sqrtm, inv

def sym(w):
	return w.dot(inv(sqrtm(w.T.dot(w))))

if len(sys.argv) < 2 or len(sys.argv) > 3:
	print "Single file: %s [regressors.csv]" % (str(sys.argv[0]))
	print "Or two files: %s [predictor.csv] [covariates.csv]" % (str(sys.argv[0]))
elif len(sys.argv) == 2:
	cmdargs = str(sys.argv)
	arg_regressors = str(sys.argv[1])
	regressors_nocsv = arg_regressors.split('.csv',1)[0]
	regressors = np.genfromtxt(arg_regressors, delimiter=',')
	regressor_ones = sm.add_constant(regressors)
	regressors_orthog = sym(regressor_ones)
	out_regressors = regressors_orthog[:,1:] 
	np.savetxt("%s_orthogonal.csv" % (regressors_nocsv), out_regressors, delimiter=",")
else:
	cmdargs = str(sys.argv)
	arg_pred = str(sys.argv[1])
	arg_covars = str(sys.argv[2])
	pred_nocsv = arg_pred.split('.csv',1)[0]
	covars_nocsv = arg_covars.split('.csv',1)[0]
	pred = np.genfromtxt(arg_pred, delimiter=',')
	pred_ones = sm.add_constant(pred)
	covars = np.genfromtxt(arg_covars, delimiter=',')
	regressor_ones=np.hstack([pred_ones,covars])
	regressors_orthog = sym(regressor_ones)
	out_regressors = regressors_orthog[:,1:]
	out_pred = out_regressors[:,:-(covars.shape[1])]
	out_covars = regressors_orthog[:,pred_ones.shape[1]:]
	np.savetxt("%s_orthogonal.csv" % (pred_nocsv), out_pred, delimiter=",")
	np.savetxt("%s_orthogonal.csv" % (covars_nocsv), out_covars, delimiter=",")
