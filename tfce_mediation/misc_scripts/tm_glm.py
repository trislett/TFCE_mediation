#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import argparse as ap
from scipy.stats import t,f
from tfce_mediation.pyfunc import dummy_code, column_product
from tfce_mediation.cynumstats import cy_lin_lstsqr_mat_residual, se_of_slope

def dummy_code(variable, iscontinous = False, demean = True):
	"""
	Dummy codes a variable
	
	Parameters
	----------
	variable : array
		1D array variable of any type 

	Returns
	---------
	dummy_vars : array
		dummy coded array of shape [(# subjects), (unique variables - 1)]
	
	"""
	if iscontinous:
		dummy_vars = variable - np.mean(variable,0)
	else:
		unique_vars = np.unique(variable)
		dummy_vars = []
		for var in unique_vars:
			temp_var = np.zeros((len(variable)))
			temp_var[variable == var] = 1
			dummy_vars.append(temp_var)
		dummy_vars = np.array(dummy_vars)[1:] # remove the first column as reference variable
		dummy_vars = np.squeeze(dummy_vars).astype(np.int).T
		if demean:
			dummy_vars = dummy_vars - np.mean(dummy_vars,0)
	return dummy_vars

def column_product(arr1, arr2):
	"""
	Multiply two dummy codes arrays
	
	Parameters
	----------
	arr1 : array
		2D array variable dummy coded array (nlength, nvars)

	arr2 : array
		2D array variable dummy coded array (nlength, nvars)

	Returns
	---------
	prod_arr : array
		dummy coded array [nlength, nvars(arr1)*nvars(arr2)]
	
	"""
	l1 = len(arr1)
	l2 = len(arr2)
	if l1 == l2:
		arr1 = np.array(arr1)
		arr2 = np.array(arr2)
		prod_arr = []
		if arr1.ndim == 1:
			prod_arr = (arr1*arr2.T).T
		elif arr2.ndim == 1:
			prod_arr = (arr2*arr1.T).T
		else:
			for i in range(arr1.shape[1]):
				prod_arr.append((arr1[:,i]*arr2.T).T)
			prod_arr = np.array(prod_arr)
			if prod_arr.ndim == 3:
				prod_arr = np.concatenate(prod_arr, axis=1)
		return prod_arr
	else:
		print("Error: Array must be of same length")
		quit()

# Type I Sum of Squares (order matters!!!)
def glm_typeI(endog, exog, dmy_covariates = None, output_fvalues = True, output_tvalues = False, output_pvalues = False):
	"""
	Generalized ANCOVA using Type I Sum of Squares
	
	Parameters
	----------
	endog : array
		Endogenous (dependent) variable array (Nsubjects, Nvariables)
	exog : array
		Exogenous (independent) dummy coded variables
		exog is an array of arrays (Nvariables, Nsubjects, Kvariable)
	dmy_covariates : array
		Dummy coded array of covariates of no interest
	
	Returns
	---------
	To-do
	"""
	
	n = endog.shape[0]
	
	kvars = []
	exog_vars = np.ones((n))
	for var in exog:
		var = np.array(var)
		if var.ndim == 1:
			kvars.append((1))
		else:
			kvars.append((var.shape[1]))
		exog_vars = np.column_stack((exog_vars,var))
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covariates))
	exog_vars = np.array(exog_vars)

	k = exog_vars.shape[1]

	DF_Between = k - 1 # aka df model
	DF_Within = n - k # aka df residuals
	DF_Total = n - 1

	SS_Total = np.sum((endog - np.mean(endog,0))**2,0)
	a, SS_Residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog)

	if output_fvalues:
		SS_Between = SS_Total - SS_Residuals
		MS_Residuals = (SS_Residuals/DF_Within)
		Fvalues = (SS_Between/DF_Between) / MS_Residuals

		# F value for exog
		Fvar = []
		Pvar = []
		start = 1
		for i,col in enumerate(kvars):
			stop = start + col
			SS_model = np.array(SS_Total - cy_lin_lstsqr_mat_residual(np.delete(exog_vars,np.s_[start:stop],1),endog)[1])
			Fvar.append((SS_Between - SS_model)/MS_Residuals*kvars[i])
			start += col
			if output_pvalues:
				Pvar.append(f.sf(Fvar[i],col,DF_Within))
	if output_tvalues:
		sigma2 = np.sum((endog - np.dot(exog_vars,a))**2,axis=0) / (n - k)
		invXX = np.linalg.inv(np.dot(exog_vars.T, exog_vars))
		if endog.ndim == 1:
			se = np.sqrt(np.diag(sigma2 * invXX))
		else:
			num_depv = endog.shape[1]
			se = se_of_slope(num_depv,invXX,sigma2,k)
		Tvalues = a / se
	# return values
	if output_tvalues and output_fvalues:
		if output_pvalues:
			Pmodel = f.sf(Fvalues,DF_Between,DF_Within)
			Pvalues = t.sf(np.abs(Tvalues), DF_Total)*2
			return (Fvalues, np.array(Fvar), Tvalues, Pmodel, np.array(Pvar), Pvalues)
		else:
			return (Fvalues, np.array(Fvar), Tvalues)
	elif output_tvalues:
		if output_pvalues:
			Pvalues = t.sf(np.abs(Tvalues), DF_Total)*2
			return (Tvalues, Pvalues)
		else:
			return Tvalues
	elif output_fvalues:
		if output_pvalues:
			Pmodel = f.sf(Fvalues,DF_Between,DF_Within)
			return (Fvalues, np.array(Fvar), Pmodel, np.array(Pvar))
		else:
			return (Fvalues, np.array(Fvar))
	else:
		print("No output has been selected")





