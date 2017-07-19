#!/usr/bin/env python

#	tm_maths: math functions for vertex and voxel images
#	Copyright (C) 2016 Tristram Lett

#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.

#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.

#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import os
import sys
import numpy as np
import argparse as ap
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA, FactorAnalysis, IncrementalPCA
from sklearn.model_selection import cross_val_score
from scipy import stats, signal

from tfce_mediation.pyfunc import zscaler
from tfce_mediation.tm_io import read_tm_filetype, write_tm_filetype, savenifti_v2, savemgh_v2

DESCRIPTION="ICA"

def tmi_run_ica(img_data_trunc, num_comp, masking_array, affine_array, variance_threshold = 0.9, timeplot = False, timeplot_name=None, filetype='nii.gz', outname='ica.nii.gz'):
	ica = FastICA(n_components=int(num_comp),max_iter=1000, tol=0.00001)
	S_ = ica.fit_transform(img_data_trunc).T
	components = ica.components_.T
	#scaling
	fitcomps = np.copy(S_)
	fitcomps = zscaler(fitcomps)
	img_data_trunc =  np.copy(fitcomps.T) # ram shouldn't be an issue here...
	np.savetxt("ICA_fit.csv", zscaler(components), fmt='%10.8f', delimiter=',')

	# variance explained.
	explained_total_var = np.zeros((int(num_comp)))
	explained_var_ratio = np.zeros((int(num_comp)))
	# total variance
	back_projection = ica.inverse_transform(S_.T)
	total_var = back_projection.var()
	for i in range(int(num_comp)):
		tempcomps = np.copy(S_)
		tempcomps[i,:] = 0
		temp_back_proj = ica.inverse_transform(tempcomps.T)
		temp_var = temp_back_proj.var()
		explained_var_ratio[i] = total_var - temp_var
		explained_total_var[i] = (total_var - temp_var) / total_var
		print "ICA # %d; Percent of Total Variance %1.3f" % ((i+1),explained_total_var[i]*100)
	explained_var_ratio = explained_var_ratio / explained_var_ratio.sum()

	sum_total_variance_explained = explained_total_var.sum()
	print "Total variance explained by all components = %1.3f" % sum_total_variance_explained
	print "Re-ordering components"
	sort_mask = (-1*explained_total_var).argsort()
	if sum_total_variance_explained > variance_threshold:
		#sort data
		sort_mask = (-1*explained_total_var).argsort()
		np.savetxt("ICA_total_var.csv",explained_total_var[sort_mask], fmt='%1.5f', delimiter=',')
		np.savetxt("ICA_explained_var_ratio.csv",explained_var_ratio[sort_mask], fmt='%1.5f', delimiter=',')
		img_data_trunc=img_data_trunc[:,sort_mask]
		if filetype=='nii.gz':
			savenifti_v2(img_data_trunc, masking_array[0], outname, affine_array[0])
		else:
			pointer = 0
			position_array = [0]
			for i in range(len(masking_array)):
				pointer += len(masking_array[i][masking_array[i]==True])
				position_array.append(pointer)
			del pointer
			for i in range(len(masking_array)):
				start = position_array[i]
				end = position_array[i+1]
				savemgh_v2(img_data_trunc[start:end], masking_array[i], "%d_%s" % (i,outname), affine_array[i])

		# save outputs and ica functions for potential ica removal
		if os.path.exists('ICA_temp'):
			print 'ICA_temp directory exists'
			exit()
		else:
			os.makedirs('ICA_temp')
		np.save('ICA_temp/signals.npy',S_)
		pickle.dump( ica, open( "ICA_temp/icasave.p", "wb" ) )


	return ica, sort_mask,  sum_total_variance_explained

def getArgumentParser(ap = ap.ArgumentParser(description = DESCRIPTION)):
	ap.add_argument("-i_tmi", "--tmifile",
		help="Input the *.tmi file for analysis.", 
		nargs=1,
		metavar=('*.tmi'),
		required=True)
	ap.add_argument("--fastica",
		help="Independent component analysis. Input the number of components (e.g.,--fastica 8 for eight components). Outputs the recovered sources, and the component fit for each subject.",
		action = 'store_true')
	ap.add_argument("--pca",
		help="PCA.",
		action = 'store_true')
	ap.add_argument("-mc", "--maxnpcacomponents",
		help="Limit the maximum components.",
		nargs=1,
		metavar=('INT'))
	ap.add_argument("-nic", "--numicacomponents",
		nargs=1,
		metavar=('INT'))
	ap.add_argument("--timeplot",
		help="Generates a figure of the components over time (or subjects) as jpeg. Input a basename for the analysis (e.g.,--timeplot lh_ica_area_plots).",
		nargs=1,
		metavar=('string'))
	ap.add_argument("--outtype", 
		help="Specify the output file type", 
		nargs=1, 
		default=['nii.gz'], 
		choices=('tmi', 'mgh', 'nii.gz'))
	ap.add_argument("-o", "--outname",
		help="specify output name",
		nargs=1)
	ap.add_argument("--detrend", 
		help="Removes the linear trend from time series data.",
		action='store_true')

	return ap

def run(opts):
	element, image_array, masking_array, maskname, affine_array, _, _, surfname, _, _, _  = read_tm_filetype(opts.tmifile[0])
	img_data_trunc = image_array[0]
	del image_array # reduce ram usage
	img_data_trunc = img_data_trunc.astype(np.float32)
	img_data_trunc[np.isnan(img_data_trunc)]=0
	if opts.scale:
		img_data_trunc = zscaler(img_data_trunc)
	if opts.maxnpcacomponents:
		numpcacomps = opts.maxnpcacomponents=[0]
	else:
		numpcacomps = img_data_trunc.shape[1]
	if opts.pca:
		pca = PCA(n_components=numpcacomps)
		S_ = pca.fit_transform(img_data_trunc).T
		for i in range(len(pca.explained_variance_ratio_)):
			if (pca.explained_variance_ratio_[i] < 0.01):
				start_comp_number = i
				print "Component %d explains %1.4f of the variance." % (i, pca.explained_variance_ratio_[0:i].sum())
				if pca.explained_variance_ratio_[0:i].sum() < 0.80:
					pass
				else:
					break

		print "%d number of components, explaining %1.2f of the variance." % (start_comp_number, (pca.explained_variance_ratio_[0:start_comp_number].sum()*100))

		rsquare_scores = []
		std_err = []
		w_rsquare_scores = []
		range_comp = np.arange(0,numpcacomps-2, 1)
		for comp_number in range_comp:
			x = np.array(range(len(pca.explained_variance_ratio_))[comp_number:])
			y = pca.explained_variance_ratio_[comp_number:]
			slope, intercept, r_value, p_value, se = stats.linregress(x,y)
			rsquare_scores.append((r_value**2))
			std_err.append((se))
			w_rsquare_scores.append((r_value**2 * ((range_comp[-1] - comp_number+1) /range_comp[-1])))

		best_comp = np.argmax(rsquare_scores)
		best_comp2 = np.argmin(std_err)
		best_comp3 = np.argmax(w_rsquare_scores)
		print "Best Component %d; R-square residual score %1.4f; variance explained %1.4f" % (best_comp, rsquare_scores[best_comp], pca.explained_variance_ratio_[:best_comp].sum())

		x = np.array(range(len(pca.explained_variance_ratio_))[best_comp:])
		y = pca.explained_variance_ratio_[best_comp:]
		m,b = np.polyfit(x,y,1)

		%matplotlib

		xaxis = np.arange(pca.explained_variance_ratio_.shape[0]) + 1
		plt.plot(xaxis, pca.explained_variance_ratio_, 'ro-', linewidth=2)

		plt.axvline(best_comp, color='r', linestyle='dashed', linewidth=2)
		plt.text(int(best_comp - 10),pca.explained_variance_ratio_.max()*.99,'Noise; Comp=%d, Sum V(e)=%1.2f' % (best_comp, pca.explained_variance_ratio_[:best_comp].sum()),rotation=90)

		plt.axvline(start_comp_number, color='g', linestyle='dashed', linewidth=2)
		plt.text(int(start_comp_number - 10),pca.explained_variance_ratio_.max()*.99,'Threshold; Comp=%d, Sum V(e)=%1.2f' % (start_comp_number, pca.explained_variance_ratio_[:start_comp_number].sum()),rotation=90)

		plt.axvline(best_comp3, color='b', linestyle='dashed', linewidth=2)
		plt.text(int(best_comp3 - 10),pca.explained_variance_ratio_.max()*.99,'Weight Noise; Comp=%d, Sum V(e)=%1.2f' % (best_comp3, pca.explained_variance_ratio_[:best_comp3].sum()),rotation=90)

		plt.plot(xaxis, m*xaxis + b, '--')
		plt.title('Scree Plot')
		plt.xlabel('Principal Component')
		plt.ylabel('Explained Variance Ratio')
		plt.show()


###### TEST ##########

#	xaxis = np.arange(pca.explained_variance_ratio_.shape[0]) + 1
#	plt.plot(xaxis, pca.explained_variance_ratio_, 'ro-', linewidth=2)
#	plt.axvline(start_comp_number, color='b', linestyle='dashed', linewidth=2)
#	plt.title('Scree Plot')
#	plt.xlabel('Principal Component')
#	plt.ylabel('Explained Variance Ratio')
#	plt.show()



	if opts.fastica:
		if opts.pca:
			num_comp = start_comp_number
		elif opts.numicacomponents:
			num_comp = int(opts.numicacomponents[0])
		else:
			print "unknown number of compenents"
			exit()
		print num_comp
		ica, sort_mask, _ = tmi_run_ica(img_data_trunc,num_comp, variance_threshold=.8, masking_array = masking_array, affine_array = affine_array, filetype='mgh', outname='ica.mgh')
		components = ica.components_.T

	if opts.timeplot:
		# generate graphs
		analysis_name = opts.timeplot[0]
#			components = np.copy(fitcomps)
		components = zscaler(components[:,sort_mask].T).T
		subs=np.array(range(components.shape[0]))+1
		time_step = 1 / 100

		if os.path.exists(analysis_name):
			print '%s directory exists' % analysis_name
			exit()
		else:
			os.makedirs(analysis_name)
		plt.figure(figsize=(10,5))
		for i in range(components.shape[1]):
			plt.plot(subs, components[:,i], 'ro-', linewidth=2)
			plt.title('Component %d Plot' % (i+1))
			plt.xlabel('Time or Subject (units)')
			plt.savefig('%s/%s_timeplot_comp%d.jpg' % (analysis_name,analysis_name,(i+1)))
			plt.clf()

			ps = np.abs(np.fft.fft(components[:,i]))**2
			freqs = np.fft.fftfreq(components[:,i].size, time_step)
			idx = np.argsort(freqs)
			plt.plot(np.abs(freqs[idx]), ps[idx])

			plt.title('Component %d Powerspectrum' % (i+1))
			plt.xlabel('Unit Frequency (Hz / 100)')
			plt.savefig('%s/%s_power_comp%d.jpg' % (analysis_name,analysis_name,(i+1)))
			plt.clf()


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)



