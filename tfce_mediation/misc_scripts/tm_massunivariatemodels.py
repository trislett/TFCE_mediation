#!/usr/bin/env python

import os
import sys
import warnings
import numpy as np
import pandas as pd
import argparse as ap
import statsmodels.api as sm

from joblib import Parallel, delayed
from time import time
from patsy import dmatrix
from scipy.stats import t, norm
from statsmodels.stats.multitest import multipletests
from tfce_mediation.cynumstats import tval_int, cy_lin_lstsqr_mat, se_of_slope

#naughty
if not sys.warnoptions:
	warnings.simplefilter("ignore")

def run_mm(trunc_data, out_data_array, exog_vars, groupVar, i):
	print i
	try:
		out_data_array = sm.MixedLM(trunc_data, exog_vars, groupVar).fit().resid
	except ValueError:
		print("Error %d" % i)
		out_data_array = np.zeros((len(exog_vars)))
	return out_data_array

def mixedmodelparallel(data_array, exog_vars, groupVar, numV, num_cores):
	resid_data = np.zeros(data_array.shape)
	resid_data = Parallel(n_jobs=num_cores)(delayed(run_mm)((data_array[i,:]+1),resid_data[i,:], exog_vars, groupVar, i) for i in range(numV))
	return np.array(resid_data)

def scalearr(X, demean = True, unitvariance = True):
	if demean:
		X -= np.mean(X, axis=0)
	if unitvariance:
		X /= np.std(X, axis=0)
	return X

def create_exog_mat(var_arr, pdDF, scale = False, scale_groups = None):
	exog_vars = pdDF['%s' % var_arr[0]]
	if len(var_arr) is not 1:
		for preds in range(len(var_arr)-1):
			exog_vars = np.column_stack((exog_vars,pdDF['%s' % var_arr[preds+1]]))
	if scale:
		print('Scaling exogenous variables')
		exog_vars = scalearr(exog_vars)
	if scale_groups is not None:
		print('Scaling exogenous variables within group')
		for group in np.unique(scale_groups):
			exog_vars[scale_groups==group] = scalearr(exog_vars[scale_groups==group])
	return np.array(sm.add_constant(exog_vars))

def russiandolls(group_list, pdDF): # mm assumes equal variances... 
	if len(group_list) == 1:
		print('You should not be here...')
	for i, groupvar in enumerate(group_list):
		if i == 0:
			pdDF['group_list'] = pdDF[group_list[0]].astype(np.str) + "_" +  pdDF[group_list[1]].astype(np.str)
		elif i == 1:
			pass
		else:
			pdDF['group_list'] = pdDF['group_list'].astype(np.str) + "_" +  pdDF[groupvar].astype(np.str)
	return pdDF

# pdCSV[groupVar[0]].isnull()
def omitmissing(pdDF, endog_range, exogenous = None, groups = None):
	isnull_arr = pdDF[pdDF.columns[endog_range[0]]].isnull() * 1
	for i in range(int(endog_range[0]),int(endog_range[1])):
		isnull_arr = np.column_stack((isnull_arr, pdDF[pdDF.columns[int(i+1)]].isnull() * 1))
	if exogenous is not None:
		if exogenous.ndim ==1:
			isnull_arr = np.column_stack((isnull_arr, np.isnan(exogenous)*1))
		else:
			for j in range(exogenous.shape[1]):
				isnull_arr = np.column_stack((isnull_arr, np.isnan(exogenous[:,j])*1))
	if groups is not None:
		for groupvar in groups:
			isnull_arr = np.column_stack((isnull_arr, pdDF[groupvar].isnull() * 1))
	sum_arr = np.sum(isnull_arr, axis=1)
	print("%d out of %d rows contains no NaNs." % (sum_arr[sum_arr==0].shape[0], sum_arr.shape[0]))
	return pdDF[sum_arr == 0]

def special_calc_sobelz(ta, tb, alg = "aroian"):
	if alg == 'aroian':
		#Aroian variant
		SobelZ = 1/np.sqrt((1/(tb**2))+(1/(ta**2))+(1/(ta**2*tb**2)))
	elif alg == 'sobel':
		#Sobel variant
		SobelZ = 1/np.sqrt((1/(tb**2))+(1/(ta**2)))
	elif alg == 'goodman':
		#Goodman variant
		SobelZ = 1/np.sqrt((1/(tb**2))+(1/(ta**2))-(1/(ta**2*tb**2)))
	else:
		print("Unknown indirect test algorithm")
		exit()
	return SobelZ

def strip_ones(arr): # WHYYYYYYY?
	if arr.shape[1] == 2:
		return arr[:,1]
	else:
		return arr[:,1:]

def find_nearest(array, value, p_array):
	idx = np.searchsorted(array, value, side="left")
	idx[idx == len(p_array)] = idx[idx == len(p_array)] - 1 # P cannot be zero...
	return p_array[idx]

def equal_lengths(length_list):
	return length_list[1:] == length_list[:-1]

def plot_residuals(residual, fitted, basename, outdir=None, scale=True):
	import matplotlib.pyplot as plt
	from statsmodels.nonparametric.smoothers_lowess import lowess

	if scale:
		residual = scalearr(residual)
		fitted = scalearr(fitted)

	fig, ax = plt.subplots()
	ax.scatter(fitted, residual)

	ys = lowess(residual, fitted)[:,1]
	ax.plot(np.sort(fitted), ys, 'r--')
	ax.set(xlabel='Fitted values', ylabel='Residuals', title=basename)
	ax.axhline(0, color='black', linestyle=':')
	if outdir is not None:
		fig.savefig("%s/resid_plot_%s.png" % (outdir, basename))
	else:
		fig.savefig("resid_plot_%s.png" % basename)
	plt.clf()


def PCAwhiten(X):
	from sklearn.decomposition import PCA
	pca = PCA(whiten=True)
	return (pca.fit_transform(X))

def ZCAwhiten(X):
	U, s, Vt = np.linalg.svd(X, full_matrices=False)
	return np.dot(U, Vt)

def orthog_columns(arr, norm = False): # N x Exog
	arr = np.array(arr, dtype=np.float32)
	out_arr = []
	for column in range(arr.shape[1]):
		if norm:
			X = sm.add_constant(scalearr(np.delete(arr,column,1)))
			y = scalearr(arr[:,column])
		else:
			X = sm.add_constant(np.delete(arr,column,1))
			y = arr[:,column]
		a = cy_lin_lstsqr_mat(X, y)
		out_arr.append(y - np.dot(X,a))
	return np.array(out_arr).T

def ortho_neareast(w):
	return w.dot(inv(sqrtm(w.T.dot(w))))

def gram_schmidt_orthonorm(X, columns=True):
	if columns:
		Q, _ = np.linalg.qr(X)
	else:
		Q, _ = np.linalg.qr(X.T)
		Q = Q.T
	return -Q

def full_glm_results(endog_arr, exog_vars, return_resids = False, only_tvals = False, PCA_whiten = False, ZCA_whiten = False,  orthogonalize = True, orthogNear = False, orthog_GramSchmidt = False):
	if np.mean(exog_vars[:,0])!=1:
		print("Warning: the intercept is not included as the first column in your exogenous variable array")
	n, num_depv = endog_arr.shape
	k = exog_vars.shape[1]

	if orthogonalize:
		exog_vars = sm.add_constant(orthog_columns(exog_vars[:,1:]))
	elif orthogNear:
		exog_vars = sm.add_constant(ortho_neareast(exog_vars[:,1:]))
	elif orthog_GramSchmidt: # for when order matters AKA type 2 sum of squares
		exog_vars = sm.add_constant(gram_schmidt_orthonorm(exog_vars[:,1:]))
	else:
		pass

	invXX = np.linalg.inv(np.dot(exog_vars.T, exog_vars))

	DFbetween = k - 1 # aka df model
	DFwithin = n - k # aka df residuals
	DFtotal = n - 1
	if PCA_whiten:
		endog_arr = PCAwhiten(endog_arr)
	if ZCA_whiten:
		endog_arr = ZCAwhiten(endog_arr)

	a = cy_lin_lstsqr_mat(exog_vars, endog_arr)
	sigma2 = np.sum((endog_arr - np.dot(exog_vars,a))**2,axis=0) / (n - k)
	se = se_of_slope(num_depv,invXX,sigma2,k)

	if only_tvals:
		return a / se
	else:
		resids = endog_arr - np.dot(exog_vars,a)
		RSS = np.sum(resids**2,axis=0)
		TSS = np.sum((endog_arr - np.mean(endog_arr, axis =0))**2, axis = 0)
		R2 = 1 - (RSS/TSS)

		std_y = np.sqrt(TSS/DFtotal)
		R2_adj = 1 - ((1-R2)*DFtotal/(DFwithin))
		Fvalues = ((TSS-RSS)/(DFbetween))/(RSS/DFwithin)
		Tvalues = a / se
		Pvalues = t.sf(np.abs(Tvalues), DFtotal)*2
		if return_resids:
			fitted = np.dot(exog_vars, a)
			return (Fvalues, Tvalues, Pvalues, R2, R2_adj, np.array(resids), np.array(fitted))
		else:
			return (Fvalues, Tvalues, Pvalues, R2, R2_adj)

def run_permutations(endog_arr, exog_vars, num_perm, stat_arr, uniq_groups = None, matched_blocks = False, return_permutations = False):
	stime = time()
	print("The accuracy is p = 0.05 +/- %.4f" % (2*(np.sqrt(0.05*0.95/num_perm))))
	np.random.seed(int(1000+time()))

	n, num_depv = endog_arr.shape
	k = exog_vars.shape[1]

	maxT_arr = np.zeros((int(k), num_perm))

	if uniq_groups is not None:
		unique_blocks = np.unique(uniq_groups)

	for i in xrange(num_perm):
		if i % 500 == 0:
			print ("%d/%d" % (i,num_perm))
		if uniq_groups is not None:
			index_groups = np.array(range(n))
			len_list = []
			for block in unique_blocks:
				len_list.append(len(uniq_groups[uniq_groups == block]))
				s = len(index_groups[uniq_groups == block])
				index_temp = index_groups[uniq_groups == block]
				index_groups[uniq_groups == block] = index_temp[np.random.permutation(s)]
			if equal_lengths(len_list): # nested blocks (add specified models!)
				mixed_blocks = np.random.permutation(np.random.permutation(len_list))
				rotate_groups = []
				for m in mixed_blocks:
					rotate_groups.append(index_groups[uniq_groups == unique_blocks[m]])
				index_groups = np.array(rotate_groups).flatten()
			nx = exog_vars[index_groups]
		else:
			nx = exog_vars[np.random.permutation(list(range(n)))]
		perm_tvalues = full_glm_results(endog_arr, nx, only_tvals=True)#[1:,:]
		perm_tvalues[np.isnan(perm_tvalues)]=0
		maxT_arr[:,i] = perm_tvalues.max(axis=1)
	corrP_arr = np.zeros_like(stat_arr)
	p_array=np.zeros(num_perm)

	for j in range(num_perm):
		p_array[j] = np.true_divide(j,num_perm)
	for k in range(maxT_arr.shape[0]):
		sorted_maxT = np.sort(maxT_arr[k,:])
		sorted_maxT[sorted_maxT<0]=0
		corrP_arr[k,:] = find_nearest(sorted_maxT,np.abs(stat_arr[k,:]),p_array)
	print("%d permutations took %1.2f seconds." % (num_perm ,(time() - stime)))
	if return_permutations:
		np.savetxt('MaxPermutedValues.csv', maxT_arr.T, delimiter=',')
		return (1 - corrP_arr)
	else:
		return (1 - corrP_arr)

def run_permutations_med(endog_arr, exog_vars, medtype, leftvar, rightvar, num_perm, stat_arr, uniq_groups = None, matched_blocks = False, return_permutations = False):
	stime = time()
	print("The accuracy is p = 0.05 +/- %.4f" % (2*(np.sqrt(0.05*0.95/num_perm))))
	np.random.seed(int(1000+time()))
	maxT_arr = np.zeros((num_perm))

	n, num_depv = endog_arr.shape
	k = exog_vars.shape[1]

	if uniq_groups is not None:
		unique_blocks = np.unique(uniq_groups)

	for i in xrange(num_perm):
		if i % 500 == 0:
			print ("%d/%d" % (i,num_perm))
		if uniq_groups is not None:
			index_groups = np.array(range(n))
			len_list = []
			for block in unique_blocks:
				len_list.append(len(uniq_groups[uniq_groups == block]))
				s = len(index_groups[uniq_groups == block])
				index_temp = index_groups[uniq_groups == block]
				index_groups[uniq_groups == block] = index_temp[np.random.permutation(s)]
			if equal_lengths(len_list): # nested blocks (add specified models!)
				mixed_blocks = np.random.permutation(np.random.permutation(len_list))
				rotate_groups = []
				for m in mixed_blocks:
					rotate_groups.append(index_groups[uniq_groups == unique_blocks[m]])
				index_groups = np.array(rotate_groups).flatten()
		else:
			index_groups = np.random.permutation(list(range(n)))

		if medtype == 'I':
			EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))

			EXOG_A = EXOG_A[index_groups]

			EXOG_B = np.column_stack((leftvar, rightvar))
			EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))

			#pathA
			t_valuesA = full_glm_results(endog_arr, EXOG_A, only_tvals=True)[1,:]
			#pathB
			t_valuesB = full_glm_results(endog_arr, EXOG_B, only_tvals=True)[1,:]

		elif medtype == 'M':
			EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))

			EXOG_A = EXOG_A[index_groups]

			EXOG_B = np.column_stack((rightvar, leftvar))
			EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))

			#pathA
			t_valuesA = full_glm_results(endog_arr, EXOG_A, only_tvals=True)[1,:]
			#pathB
			t_valuesB = full_glm_results(endog_arr, EXOG_B, only_tvals=True)[1,:]

		elif medtype == 'Y':
			EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))
			EXOG_B = np.column_stack((rightvar, leftvar))
			EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))

			EXOG_A = EXOG_A[index_groups]
			EXOG_B = EXOG_B[index_groups]

			#pathA
			t_valuesA = sm.OLS(rightvar, EXOG_A).fit().tvalues[1]
			#pathB
			t_valuesB = full_glm_results(endog_arr, EXOG_B, only_tvals=True)[1,:]

		perm_zvalues  = special_calc_sobelz(np.array(t_valuesA), np.array(t_valuesB), alg = "aroian")
		perm_zvalues[np.isnan(perm_zvalues)]=0
		maxT_arr[i] = perm_zvalues.max()
	corrP_arr = np.zeros_like(stat_arr)
	p_array=np.zeros(num_perm)
	for j in range(num_perm):
		p_array[j] = np.true_divide(j,num_perm)
	sorted_maxT = np.sort(maxT_arr[:])
	sorted_maxT[sorted_maxT<0]=0
	corrP_arr[:] = find_nearest(sorted_maxT,np.abs(stat_arr[:]),p_array)
	print("%d permutations took %1.2f seconds." % (num_perm ,(time() - stime)))
	if return_permutations:
		np.savetxt('MaxPermutedValues.csv', maxT_arr.T, delimiter=',')
	return (1 - corrP_arr)


DESCRIPTION = 'Run linear- and linear-mixed models for now.'

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-m", "--statsmodel",
		help="Select the statistical model.",
		choices=['mixedmodel', 'mm', 'linear', 'lm'],
		required=False)
	ap.add_argument("-i", "-i_csv", "--inputcsv",
		help="Edit existing *.tmi file.",
		nargs='+', 
		metavar='*.csv',
		required=True)
	ap.add_argument("-o", "--outstats",
		help="Save stats as a *.csv file.",
		nargs=1,
		metavar='str')
	ap.add_argument("-s_csv", "--savecsv",
		help="Save the merged *.csv file.",
		nargs=1, 
		metavar='*.csv',
		required=False)
	ap.add_argument("-ic", "--indexcolumns",
		help="Select the index column for merging *.csv files. Each value in the column should be unique. Default: %(default)s)",
		nargs=1,
		default=['SubjID'])
	ap.add_argument("-on", "--outputcolumnnames",
		help="Output column names and exits",
		action='store_true')
	ap.add_argument("-exog", "-iv","--exogenousvariables",
		help="Exogenous (independent) variables. Intercept(s) will be included automatically. e.g. -exog {time_h} {weight_lbs}",
		metavar='str',
		nargs='+')
	ap.add_argument("-int", "--interactionvars",
		help="Interaction exogenous (independent) variables. Variables should not be included in -exog. e.g. -int diagnosis*genescore sex*age",
		metavar = ['Var1*Var2'],
		nargs = '+')
	ap.add_argument("-g", "--groupingvariable",
		help="Select grouping variable for mixed model.",
		metavar='str',
		nargs='+')
	ap.add_argument("-ei", "--exogintercept",
		help="Select the intercept for the grouping variable for the linear mixed model. -gi {subject}",
		metavar='str',
		nargs=1)
	ap.add_argument("-med", "--mediation",
		help="Select mediation type and mediation variables. The left and right variables must be not included in exogenouse variabls! e.g., -med {medtype(I|M|Y)} {leftvar} {rightvar}",
		nargs=3,
		metavar=('{I|M|Y}', 'LeftVar', 'RightVar'))
	ap.add_argument("-se", "--scaleexog",
		help="Scale the exogenous/independent variables",
		action='store_true')
	ap.add_argument("-seg", "--scaleexogwithingroup",
		help="Scale the exogenous/independent variables within the grouping variable",
		action='store_true')
	ap.add_argument("-r", "--range",
		help="Select the range of columns for analysis",
		nargs=2,
		metavar='int',
		type=int,
		required=False)
	ap.add_argument("-p", "--parallel",
		help="parallel computing for -sr. -p {int}",
		metavar='int',
		nargs=1)
	ap.add_argument("-rand", "--permutation",
		help="Permutation testing for FWER correction. Must be used with -lm. Block variable(s) can be specficied with -g. -rand {num_perm}",
		metavar='int',
		nargs=1)
	ap.add_argument("-pr", "--plotresids",
		help="Output residual plots for mixed models.",
		action='store_true')
	return ap

def run(opts):

	indexCol = opts.indexcolumns[0]

	# read csv(s)
	num_csv=len(opts.inputcsv)
	pdCSV = pd.read_csv(opts.inputcsv[0], delimiter=',', index_col=[indexCol])
	if num_csv > 1:
		for i in range(int(num_csv-1)):
			tempCSV = pd.read_csv(opts.inputcsv[int(i+1)], delimiter=',', index_col=[indexCol])
			pdCSV = pd.concat([pdCSV, tempCSV], axis=1, join_axes=[pdCSV.index])

	# Interaction Variables
	if opts.interactionvars:
		for int_terms in opts.interactionvars:
			inteaction_vars = int_terms.split("*")
			for scale_var in inteaction_vars:
				var_temp = scalearr(pdCSV[scale_var])
				var_tempname = '%s_std' % scale_var
				if var_tempname in opts.exogenousvariables:
					pass
				else:
					pdCSV[var_tempname] = var_temp
					opts.exogenousvariables.append(var_tempname)
		for int_terms in opts.interactionvars:
			inteaction_vars = int_terms.split("*")
			for i, scale_var in enumerate(inteaction_vars):
				if i == 0:
					int_temp = pdCSV['%s_std' % scale_var]
					int_tempname = '%s' % scale_var
				else:
					int_temp = int_temp * pdCSV['%s_std' % scale_var]
					int_tempname = int_tempname + '.X.' + scale_var
			if int_tempname in opts.exogenousvariables:
				pass
			else:
				pdCSV[int_tempname] = int_temp
				opts.exogenousvariables.append(int_tempname)
			int_temp = None
		print opts.exogenousvariables


	# output column/variable names.
	if opts.outputcolumnnames:
		for counter, roi in enumerate(pdCSV.columns):
			print("[%d] : %s" % (counter, roi))
		quit()

	# set grouping variables
	if opts.groupingvariable:
		if len(opts.groupingvariable) > 1:
			pdCSV = russiandolls(opts.groupingvariable, pdCSV)
			groupVar = 'group_list'
		else:
			groupVar = opts.groupingvariable[0]

	# stats functions

	if opts.outstats:
		if not opts.statsmodel:
			print("A statistical model must be specificed. -m {model}")
			quit()
		if not opts.range:
			print("Range must be specfied. -r {start} {stop}")
			quit()
		elif len(opts.range) != 2:
			print("Range must have start and stop. -r {start} {stop}")
			quit()
		else:
			roi_names = []
			t_values = []
			p_values = []
			icc_values = []
			if not opts.exogenousvariables:
				print("The exogenous (independent) variables must be specifice. e.g., -exog pred1 pred2 age")
				quit()

			if opts.mediation:
				medvars = ['%s' % opts.mediation[1], '%s' % opts.mediation[2]]
				exog_vars = create_exog_mat(opts.exogenousvariables, pdCSV)
				# build null array
				pdCSV = omitmissing(pdDF = pdCSV,
								endog_range = opts.range, 
								exogenous = strip_ones(exog_vars),
								groups = medvars)
				if opts.statsmodel == 'mixedmodel' or opts.statsmodel == 'mm':
					pdCSV = omitmissing(pdDF = pdCSV,
									endog_range = opts.range, 
									groups = opts.groupingvariable)
				# rebuild exog_vars with correct length
				exog_vars = create_exog_mat(opts.exogenousvariables, pdCSV, opts.scaleexog==True)
				leftvar = pdCSV[opts.mediation[1]]
				rightvar = pdCSV[opts.mediation[2]]
				y = pdCSV.iloc[:,int(opts.range[0]):int(opts.range[1])+1]

				if opts.statsmodel == 'mixedmodel' or opts.statsmodel == 'mm':
					t_valuesA = []
					t_valuesB = []
					################ MM mediation ################
					if opts.mediation[0] == 'I':
						EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))
						EXOG_B = np.column_stack((leftvar, rightvar))
						EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))
						#pathA
						for i in xrange(int(opts.range[0]),int(opts.range[1])+1):
							mdl_fit = sm.MixedLM(pdCSV[pdCSV.columns[i]], 
										EXOG_A,
										pdCSV[groupVar]).fit()
							roi_names.append(pdCSV.columns[i])
							t_valuesA.append(mdl_fit.tvalues[1])
						#pathB
						for i in xrange(int(opts.range[0]),int(opts.range[1])+1):
							mdl_fit = sm.MixedLM(pdCSV[pdCSV.columns[i]],
										EXOG_B,
										pdCSV[groupVar]).fit()
							t_valuesB.append(mdl_fit.tvalues[1])
					elif opts.mediation[0] == 'M':
						EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))
						EXOG_B = np.column_stack((rightvar, leftvar))
						EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))
						#pathA
						for i in xrange(int(opts.range[0]),int(opts.range[1])+1):
							mdl_fit = sm.MixedLM(pdCSV[pdCSV.columns[i]],
										EXOG_A,
										pdCSV[groupVar]).fit()
							roi_names.append(pdCSV.columns[i])
							t_valuesA.append(mdl_fit.tvalues[1])
						#pathB
						for i in xrange(int(opts.range[0]),int(opts.range[1])+1):
							mdl_fit = sm.MixedLM(pdCSV[pdCSV.columns[i]], EXOG_B, pdCSV[groupVar]).fit()
							t_valuesB.append(mdl_fit.tvalues[1])
					else:
						EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))
						EXOG_B = np.column_stack((rightvar, leftvar))
						EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))

						#pathA
						mdl_fit = sm.MixedLM(rightvar, EXOG_A, pdCSV[groupVar]).fit()
						t_valuesA = mdl_fit.tvalues[1]

						#pathB
						for i in xrange(int(opts.range[0]),int(opts.range[1])+1):
							mdl_fit = sm.MixedLM(pdCSV[pdCSV.columns[i]],
										exog_vars,
										pdCSV[groupVar]).fit()
							roi_names.append(pdCSV.columns[i])
							t_valuesB.append(mdl_fit.tvalues[1])


					z_values  = special_calc_sobelz(np.array(t_valuesA),
									np.array(t_valuesB),
									alg = "aroian")
					p_values = norm.sf(abs(z_values))
					p_FDR = multipletests(p_values, method = 'fdr_bh')[1]

				else:
					################ LM mediation ################
					if opts.mediation[0] == 'I':
						EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))
						EXOG_B = np.column_stack((leftvar, rightvar))
						EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))

						y = pdCSV.iloc[:,int(opts.range[0]):int(opts.range[1])+1]
						#pathA
						t_valuesA = full_glm_results(y, EXOG_A, only_tvals=True)[1,:]
						#pathB
						t_valuesB = full_glm_results(y, EXOG_B, only_tvals=True)[1,:]

					elif opts.mediation[0] == 'M':
						EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))
						EXOG_B = np.column_stack((rightvar, leftvar))
						EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))

						y = pdCSV.iloc[:,int(opts.range[0]):int(opts.range[1])+1]
						#pathA
						t_valuesA = full_glm_results(y, EXOG_A, only_tvals=True)[1,:]
						#pathB
						t_valuesB = full_glm_results(y, EXOG_B, only_tvals=True)[1,:]

					elif opts.mediation[0] == 'Y':
						EXOG_A = sm.add_constant(np.column_stack((leftvar, strip_ones(exog_vars))))
						EXOG_B = np.column_stack((rightvar, leftvar))
						EXOG_B = sm.add_constant(np.column_stack((EXOG_B, strip_ones(exog_vars))))

						y = pdCSV.iloc[:,int(opts.range[0]):int(opts.range[1])+1]
						#pathA
						t_valuesA = sm.OLS(rightvar, EXOG_A).fit().tvalues[1]
						#pathB
						t_valuesB = full_glm_results(y, EXOG_B, only_tvals=True)[1,:]

					else:
						print("Error: Invalid mediation type.")
						quit()
					z_values  = special_calc_sobelz(np.array(t_valuesA), np.array(t_valuesB), alg = "aroian")
					p_values = norm.sf(abs(z_values))
					p_FDR = multipletests(p_values, method = 'fdr_bh')[1]

					if opts.permutation:
						if opts.groupingvariable:
							p_FWER = run_permutations_med(endog_arr = y,
								exog_vars = exog_vars,
								medtype = opts.mediation[0],
								leftvar = leftvar,
								rightvar = rightvar,
								num_perm = int(opts.permutation[0]),
								stat_arr = z_values,
								uniq_groups = pdCSV[groupVar],
								return_permutations = True)
						else:
							p_FWER = run_permutations_med(endog_arr = y,
								exog_vars = exog_vars,
								medtype = opts.mediation[0],
								leftvar = leftvar,
								rightvar = rightvar,
								num_perm = int(opts.permutation[0]),
								stat_arr = z_values,
								uniq_groups = None,
								return_permutations = True)

					roi_names = []
					for i in xrange(int(opts.range[0]),int(opts.range[1])+1):
						roi_names.append(pdCSV.columns[i])

				columnnames = []
				columnnames.append('Zval')
				columnnames.append('pval')
				columnnames.append('pFDR')
				columndata = np.column_stack((z_values, p_values))
				columndata = np.column_stack((columndata, p_FDR))
				if opts.permutation:
					columnnames.append('pFWER')
					columndata = np.column_stack((columndata, p_FWER))
				pd_DF = pd.DataFrame(data=columndata, index=roi_names, columns=columnnames)
				pd_DF.to_csv(opts.outstats[0], index_label='ROI')

			else:
				################ MIXED MODEL ################
				if opts.statsmodel == 'mixedmodel' or opts.statsmodel == 'mm':
					exog_vars = create_exog_mat(opts.exogenousvariables, pdCSV)

					# build null array
					pdCSV = omitmissing(pdDF = pdCSV,
									endog_range = opts.range, 
									exogenous = strip_ones(exog_vars),
									groups = opts.groupingvariable)
					# rebuild exog_vars with correct length
					if opts.scaleexogwithingroup:
						exog_vars = create_exog_mat(opts.exogenousvariables, 
							pdCSV,
							opts.scaleexog==True,
							scale_groups = pdCSV[groupVar])
					else:
						exog_vars = create_exog_mat(opts.exogenousvariables,
							pdCSV,
							opts.scaleexog==True)

					exog_re = None
					if opts.exogintercept:
						exog_re = dmatrix("1+%s" % opts.exogintercept[0], pdCSV)

					for i in xrange(int(opts.range[0]),int(opts.range[1])+1):
						mdl_fit = sm.MixedLM(endog = pdCSV[pdCSV.columns[i]],
										exog = exog_vars,
										groups = pdCSV[groupVar],
										exog_re = exog_re).fit()
						roi_names.append(pdCSV.columns[i])
						t_values.append(mdl_fit.tvalues[1:])
						p_values.append(mdl_fit.pvalues[1:])
						icc_values.append(np.array(mdl_fit.cov_re/(mdl_fit.cov_re + mdl_fit.scale)))
						if opts.plotresids:
							os.system('mkdir -p resid_plots')
							plot_residuals(residual=mdl_fit.resid,
									fitted=mdl_fit.fittedvalues,
									basename=('%s_mm_%s' % (str(i).zfill(4), pdCSV.columns[i])),
									outdir='resid_plots/')
					p_values = np.array(p_values)
					t_values = np.array(t_values)
					p_FDR = np.zeros_like(p_values)

					p_values[np.isnan(p_values)]=1
					for col in range(p_FDR.shape[1]):
						p_FDR[:,col] = multipletests(p_values[:,col], method = 'fdr_bh')[1]


					columnnames = []
					for colname in opts.exogenousvariables:
						columnnames.append('tval_%s' % colname)
					if opts.exogintercept:
						columnnames.append('tval_re1')
						columnnames.append('tval_re1Xre2')
						columnnames.append('tval_re2')
					else:
						columnnames.append('tval_groupRE')

					for colname in opts.exogenousvariables:
						columnnames.append('pval_%s' % colname)
					if opts.exogintercept:
						columnnames.append('pval_re1')
						columnnames.append('pval_re1Xre2')
						columnnames.append('pval_re2')
					else:
						columnnames.append('pval_groupRE')

					for colname in opts.exogenousvariables:
						columnnames.append('pFDR_%s' % colname)
					if opts.exogintercept:
						columnnames.append('pFDR_re1')
						columnnames.append('pFDR_re1Xre2')
						columnnames.append('pFDR_re2')
					else:
						columnnames.append('pFDR_groupRE')

					if not opts.exogintercept:
						columnnames.append('ICC_groupRE')
					columndata = np.column_stack((t_values, p_values))
					columndata = np.column_stack((columndata, p_FDR))
					if not opts.exogintercept:
						columndata = np.column_stack((columndata, np.array(icc_values).flatten()))
					pd_DF = pd.DataFrame(data=columndata, index=roi_names, columns=columnnames)
					pd_DF.to_csv(opts.outstats[0], index_label='ROI')
				else:
					################ LINEAR MODEL ################
					exog_vars = create_exog_mat(opts.exogenousvariables, pdCSV)
					# build null array
					pdCSV = omitmissing(pdDF = pdCSV,
									endog_range = opts.range,
									exogenous = strip_ones(exog_vars))
					# rebuild exog_vars with correct length
					if opts.scaleexogwithingroup:
						exog_vars = create_exog_mat(opts.exogenousvariables, 
							pdCSV,
							opts.scaleexog==True,
							scale_groups = pdCSV[groupVar])
					else:
						exog_vars = create_exog_mat(opts.exogenousvariables,
							pdCSV,
							opts.scaleexog==True)
					y = np.array(pdCSV.iloc[:,int(opts.range[0]):int(opts.range[1])+1])

					if opts.plotresids:
						f_values, t_values, p_values, R2, R2_adj, resids, fitted = full_glm_results(y, exog_vars, return_resids = True)
					else:
						np.savetxt('temp_int.csv', orthog_columns(strip_ones(exog_vars)), delimiter=',')
						f_values, t_values, p_values, R2, R2_adj = full_glm_results(y, exog_vars)

					if opts.permutation:
						if opts.groupingvariable:
							p_FWER = run_permutations(endog_arr = y,
								exog_vars = exog_vars,
								num_perm = int(opts.permutation[0]),
								stat_arr = t_values,
								uniq_groups = pdCSV[groupVar],
								return_permutations = True)
						else:
							p_FWER = run_permutations(endog_arr = y,
								exog_vars = exog_vars,
								num_perm = int(opts.permutation[0]),
								stat_arr = t_values,
								uniq_groups = None,
								return_permutations = True)
						p_FWER = p_FWER[1:,:].T

					t_values = t_values[1:,:].T # ignore intercept
					p_values = p_values[1:,:].T # ignore intercept

					roi_names = []
					for i in xrange(int(opts.range[0]),int(opts.range[1])+1):
						roi_names.append(pdCSV.columns[i])

					p_FDR = np.zeros_like(p_values)
					p_values[np.isnan(p_values)]=1
					for col in range(p_FDR.shape[1]):
						p_FDR[:,col] = multipletests(p_values[:,col], method = 'fdr_bh')[1]

					columnnames = []
					columnnames.append('Fvalue')
					columnnames.append('R2')
					columnnames.append('R2adj')
					for colname in opts.exogenousvariables:
						columnnames.append('tval_%s' % colname)
					for colname in opts.exogenousvariables:
						columnnames.append('pval_%s' % colname)
					for colname in opts.exogenousvariables:
						columnnames.append('pFDR_%s' % colname)

					columndata = np.column_stack((f_values[:,np.newaxis], R2))
					columndata = np.column_stack((columndata, R2_adj))
					columndata = np.column_stack((columndata, t_values))
					columndata = np.column_stack((columndata, p_values))
					columndata = np.column_stack((columndata, p_FDR))
					if opts.permutation:
						for colname in opts.exogenousvariables:
							columnnames.append('pFWER_%s' % colname)
						columndata = np.column_stack((columndata, p_FWER))
					pd_DF = pd.DataFrame(data=columndata, index=roi_names, columns=columnnames)
					pd_DF.to_csv(opts.outstats[0], index_label='ROI')

					if opts.plotresids:
						os.system('mkdir -p resid_plots')
						for i, roi in enumerate(np.array(roi_names)):
							plot_residuals(residual = resids[:,i],
									fitted = fitted[:,i],
									basename=('%s_lm_%s' % (str(i+int(opts.range[0])).zfill(4), roi)),
									outdir='resid_plots/')


	if opts.savecsv:
		pdCSV.to_csv(opts.savecsv[0])

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
