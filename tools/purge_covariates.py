#!/usr/bin/python

import sys
import numpy as np
import statsmodels.api as sm

if len(sys.argv) < 3:
	print "*******************************************************************"
	print "Usage: %s [Dependent Variable CSV] [Independent Variable CSV]" % (str(sys.argv[0]))
	print "  "
	print "Outputs residual file after correcting for covariates"
	print "Both files should have the same length (i.e., number of subjects)"
	print "*******************************************************************"
else:
	cmdargs = str(sys.argv)
	arg_depvars = str(sys.argv[1])
	arg_covars = str(sys.argv[2])
	depvars_nocsv = arg_depvars.split('.csv',1)[0]
	covars_nocsv = arg_covars.split('.csv',1)[0]
	depvars = np.genfromtxt(arg_depvars, delimiter=',')
	covars = np.genfromtxt(arg_covars, delimiter=',')

	x_covars = sm.add_constant(covars)
	a_c = np.linalg.lstsq(x_covars, depvars)[0]
	resids = depvars - np.dot(x_covars,a_c)

	np.savetxt(("%s_resids.csv" % depvars_nocsv), resids, delimiter=",", fmt='%10.5f')
