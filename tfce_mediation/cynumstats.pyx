#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

#    Various statistic functions optimized in cython
#    Copyright (C) 2016  Tristram Lett

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
cimport cython

def cy_lin_lstsqr_mat(X, y):
   return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)

def calcF(X,y, n, k):
   a = cy_lin_lstsqr_mat(X, y)
   resids = y - np.dot(X,a)
   RSS = sum(resids**2)
   TSS = sum((y - np.mean(y))**2)
   return ((TSS-RSS)/(k-1))/(RSS/(n-k))

def cython_lstsqr(x, y):
   x_avg = sum(x)/len(x)
   y_avg = sum(y)/len(y)
   var_x = sum([(x_i - x_avg)**2 for x_i in x])
   cov_xy = sum([(x_i - x_avg)*(y_i - y_avg) for x_i,y_i in zip(x,y)])
   slope = cov_xy / var_x
   y_interc = y_avg - slope*x_avg
   return (y_interc,slope)

def se_of_slope(num_voxel,invXX,sigma2,k):
   cdef int j
   se = np.zeros(shape=(k,num_voxel))
   for j in xrange(num_voxel):
      se[:,j] = np.sqrt ( np.diag (sigma2[j] * invXX ) )
   return se

def resid_covars (x_covars, data):
   a_c = cy_lin_lstsqr_mat(x_covars, data.T)
   resids = data.T - np.dot(x_covars,a_c)
   return resids

def tval_int(X, invXX, y, n, k, numvoxel):
   a = cy_lin_lstsqr_mat(X, y)
   sigma2 = np.sum((y - np.dot(X,a))**2,axis=0) / (n - k)
   se = se_of_slope(numvoxel,invXX,sigma2,k)
   tvals = a / se
   return tvals

def calc_beta_se(x,y,n,num_voxel):
   X = np.column_stack([np.ones(n),x])
   invXX = np.linalg.inv(np.dot(X.T, X))
   k = len(X.T)
   a = cy_lin_lstsqr_mat(X, y)
   beta = a[1]
   sigma2 = np.sum((y - np.dot(X,a))**2,axis=0) / (n - k)
   se = se_of_slope(num_voxel,invXX,sigma2,k)
   return (beta,se)
