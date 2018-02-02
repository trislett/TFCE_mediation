#!/usr/bin/env python


import os
import numpy as np
import argparse as ap
import pickle
import nibabel as nib
from sklearn.decomposition import FastICA, LatentDirichletAllocation, MiniBatchSparsePCA, NMF, PCA
from tfce_mediation.pyfunc import loadnifti, savenifti, savemgh, zscaler, minmaxscaler
from tfce_mediation.cynumstats import resid_covars

def loadcortextmgh(lh_imagename, rh_imagename, indexl = None, indexr = None):
	if (os.path.exists(lh_imagename)) and (os.path.exists(rh_imagename)): # check if files exist
		lh_img = nib.freesurfer.mghformat.load(lh_imagename)
		lh_img_data = lh_img.get_data()
		if indexl is not None:
			lh_mask_index = indexl
		else:
			lh_mean_data = np.mean(np.abs(lh_img_data),axis=3)
			lh_mask_index = (lh_mean_data != 0)
		rh_img = nib.freesurfer.mghformat.load(rh_imagename)
		rh_img_data = rh_img.get_data()
		if indexr is not None:
			rh_mask_index = indexr
		else:
			rh_mean_data = np.mean(np.abs(rh_img_data),axis=3)
			rh_mask_index = (rh_mean_data != 0)
		lh_img_data_trunc = lh_img_data[lh_mask_index]
		rh_img_data_trunc = rh_img_data[rh_mask_index]
		img_data_trunc = np.vstack((lh_img_data_trunc,rh_img_data_trunc))
		midpoint = lh_img_data_trunc.shape[0]
	else:
		print("Cannot find input images")
		exit()
	return (img_data_trunc, midpoint, lh_img, rh_img, lh_mask_index, rh_mask_index)

def run_decomp(data, method = 'ICA', num_comp = 10, sort_comp = False, scale_results = False):
	if method == 'ICA':
		model = FastICA(n_components=int(num_comp),
			max_iter=1000,
			tol=0.00001)
	elif method == 'NMF':
		model = NMF(n_components=int(num_comp), 
			init='nndsvda')
	elif method == 'LDA':
		model = LatentDirichletAllocation(n_topics=int(num_comp), 
			n_jobs = -1, 
			learning_method = 'batch')
	elif method == 'sPCA':
		model = MiniBatchSparsePCA(n_components=int(num_comp), 
			alpha = 0.01,
			n_jobs = -1)
	elif method == 'PCA':
		model = PCA(n_components=int(num_comp))
	else:
		print("Method %s not supported" % method)
		quit()
	S_ = model.fit_transform(data).T
	components = np.zeros_like(model.components_.T)
	components[:] = np.copy(model.components_.T)
	#scaling
	fitcomps = np.zeros_like(S_)
	fitcomps[:] = np.copy(S_)
	outdata = fitcomps.T
	if sort_comp:
		# variance explained.
		explained_total_var = np.zeros((int(num_comp)))
		explained_var_ratio = np.zeros((int(num_comp)))
		# total variance
		back_projection = model.inverse_transform(S_.T)
		total_var = back_projection.var()
		for i in range(int(num_comp)):
			tempcomps = np.zeros_like(S_)
			tempcomps[:] = np.copy(S_)
			tempcomps[i,:] = 0
			temp_back_proj = model.inverse_transform(tempcomps.T)
			temp_var = temp_back_proj.var()
			explained_var_ratio[i] = total_var - temp_var
			explained_total_var[i] = (total_var - temp_var) / total_var
			print("Component # %d; Percent of Total Variance %1.3f" % ((i+1),explained_total_var[i]*100))
			tempcomps = None
			temp_back_proj = None
		explained_var_ratio = explained_var_ratio / explained_var_ratio.sum()
		sum_total_variance_explained = explained_total_var.sum()
		print("Total variance explained by all components = %1.3f" % sum_total_variance_explained)
		print("Re-ordering components")
		sort_mask = (-1*explained_total_var).argsort()
		np.savetxt("%s_total_var.csv" % method,
			explained_total_var[sort_mask],
			fmt='%1.5f',
			delimiter=',')
		np.savetxt("%s_explained_var_ratio.csv" % method,
			explained_var_ratio[sort_mask],
			fmt='%1.5f',
			delimiter=',')
		outdata = outdata[:,sort_mask]
		if scale_results:
			components = zscaler(components[:,sort_mask])
		np.savetxt("%s_fit.csv" % method, 
			components,
			fmt='%10.8f',
			delimiter=',')
		return (outdata, sort_mask, model)
	else:
		if scale_results:
			components = zscaler(components)
		np.savetxt("%s_fit.csv" % method, 
			components,
			fmt='%10.8f',
			delimiter=',')
		return (outdata, model)



def getArgumentParser(ap = ap.ArgumentParser(description = 'test')):

	ap.add_argument("-ive", "--inputvertex", 
		help="Vertex input",
		metavar=('lh.*.mgh','rh.*.mgh'),
		action = 'append',
		nargs=2)
	ap.add_argument("-ivo", "--inputvoxel", 
		help="voxel input",
		metavar=('data','mask'),
		action = 'append',
		nargs=2)
	ap.add_argument("-it","--inputtext", 
		help="C(data)xR(Subject)",
		metavar=('*.csv'),
		action = 'append',
		nargs=1)
	ap.add_argument("-m","--method", 
		help="Method choices are: %(choices)s",
		choices = ['NMF', 'LDA', 'ICA','sPCA','PCA'],
		required = True,
		nargs=1)
	ap.add_argument("-nc","--numcomp", 
		help="Number of components",
		type = int,
		default = [10],
		nargs = 1)
	ap.add_argument("-c","--covariates", 
		help="Regress out covariates of no interest",
		metavar=('*.csv'),
		nargs = 1)
	ap.add_argument("-sort","--sortcomponents", 
		help = "Sort components by variance explained (useful for ICA)",
		default = False,
		action = 'store_true')
	ap.add_argument("--savemodel", 
		help = "Save the model as binary pickle object.",
		action = 'store_true')

	ap.add_argument("-smm","--scaleminmax", 
		help="Scale using min/max scaler",
		action='store_true')
	ap.add_argument("-sz","--scalezscore", 
		help="Scaled data to have zero mean and unit variance",
		action='store_true')
	ap.add_argument("--scaleresults", 
		help="Scaled results to have zero mean and unit variance",
		default = False,
		action='store_true')
	ap.add_argument("-o","--outname", 
		help="Basename for output",
		nargs=1)
	return ap

def run(opts):
	data = None
	if opts.covariates:
		covars = np.genfromtxt(opts.covariates[0], delimiter=',')
		x_covars = np.column_stack([np.ones(len(covars)),covars])
	if opts.inputvertex:
		for i, options in enumerate(opts.inputvertex):
			if data is None:
				data, midpoint, lh_img, rh_img, lh_mask_index, rh_mask_index = loadcortextmgh(options[0], options[1])
				if opts.covariates:
					data = resid_covars(x_covars, data).T
				if opts.scaleminmax:
					data = minmaxscaler(data.T).T
				elif opts.scalezscore:
					data = zscaler(data.T).T
				else:
					pass
			else:
				temp_data, _, _, _, _, _ = loadcortextmgh(options[0], options[1], lh_mask_index, rh_mask_index)
				if opts.covariates:
					temp_data = resid_covars(x_covars, temp_data).T
				if opts.scaleminmax:
					temp_data = minmaxscaler(temp_data.T).T
				elif opts.scalezscore:
					temp_data = zscaler(temp_data.T).T
				else:
					pass
				data = np.vstack((data, temp_data))
				temp_data = None
	if opts.inputvoxel:
		mask_array = []
		affines_array = []
		hdr_array = []
		for i, options in enumerate(opts.inputvoxel):
			mask_name = options[1]
			if not os.path.isfile(mask_name):
				print('Error %s not found.' % mask_name)
				quit()
			img_mask = nib.load(mask_name)
			mask_array.append(img_mask.get_data())
			affines_array.append(img_mask.get_affine())
			hdr_array.append(img_mask.get_header())

			#check if minc file
			img_all_name = options[0]
			_, file_ext = os.path.splitext(img_all_name)
			if file_ext == '.gz':
				_, file_ext = os.path.splitext(img_all_name)
				if file_ext == '.mnc':
					imgext = '.mnc'
					img_all = nib.load(img_all_name)
				else:
					imgext = '.nii.gz'
					os.system("zcat %s > temp_4d.nii" % img_all_name)
					img_all = nib.load('temp_4d.nii')
			elif file_ext == '.nii':
				imgext = '.nii.gz' # default to zipped images
				img_all = nib.load(img_all_name)
			else:
				print('Error filetype for %s is not supported' % img_all_name)
				quit()
			if data is None:
				data = img_all[mask_array[i] != 0]
			else:
				data = np.vstack((data, img_all[mask_array[i] != 0]))
			img_all = None

	if opts.inputtext:
		for options in opts.inputtext:
			temp_data = np.genfromtxt(options[0], delimiter=',').T
			if opts.covariates:
				temp_data = resid_covars(x_covars, temp_data).T
			if temp_data.ndim == 1:
				temp_data = temp_data[np.newaxis,:]
			if opts.scaleminmax:
				temp_data = minmaxscaler(temp_data.T).T
			elif opts.scalezscore:
				temp_data = zscaler(temp_data.T).T
			else:
				pass
			if data is None:
				data = temp_data
			else:
				data = np.vstack((data, temp_data))
			temp_data = None

	if opts.sortcomponents:
		out, transform, mdl = run_decomp(data, method = str(opts.method[0]), num_comp = opts.numcomp[0], sort_comp = True, scale_results = opts.scaleresults)
		np.save('%s_sort_transform.npy' % str(opts.method[0]), transform)
	else:
		out, mdl = run_decomp(data, method = str(opts.method[0]), num_comp = opts.numcomp[0], scale_results = opts.scaleresults)
		if str(opts.method[0]) == 'PCA':
			print(mdl.explained_variance_ratio_)

	if opts.savemodel:
		pickle.dump(mdl, open( "%s_model_dump.p" % str(opts.method[0]), "wb"))

	if opts.inputvertex:
		position = 0
		for options in opts.inputvertex:
			size = lh_mask_index[lh_mask_index==True].shape[0]
			tempout = out[position:(position+size),:]
			if opts.scaleresults:
				tempout = zscaler(tempout.T).T
			savemgh(tempout,
				lh_img,
				lh_mask_index,
				'%s_%s' % (opts.method[0],options[0]))
			position += size
			size = rh_mask_index[rh_mask_index==True].shape[0]
			tempout = out[position:(position+size),:]
			if opts.scaleresults:
				tempout = zscaler(tempout.T).T
			savemgh(tempout,
				rh_img,
				rh_mask_index,
				'%s_%s' % (opts.method[0],options[1]))
			position += size

	if opts.inputtext: # lazy for now
		if not opts.inputvertex:
			np.savetxt('%s_model_fit_text.csv' % opts.method[0],
				out,
				delimiter=',')
		else:
			np.savetxt('%s_model_fit.csv' % opts.method[0],
				out[position:(position+size),:].T,
				delimiter=',')


if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
