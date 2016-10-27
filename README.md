# TFCE_mediation
Fast regression and mediation analysis of vertex or voxel MR data with TFCE

(This is a placeholder for the full instructions.)

### Dependencies ###

TFCE_mediation should work with any UNIX based systemt. It has been tested on Ubuntu 14.04 and 16.04, Arch Linux, and OSX   
   
Required   
   
* [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki) (for vertex-based analyses)

Recommended

* [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki)
* [Neurodebian reprository](http://neuro.debian.net/)
* [PIP](https://pip.pypa.io/en/stable/installing/)

Recommended for parallelization:

* [GNU Parallel](http://www.gnu.org/software/parallel/)

or

* For personal installation of [HTCondor](https://research.cs.wisc.edu/htcondor/) (i.e., Condor), you can follow the instructions [here](http://neuro.debian.net/blog/2012/2012-03-09_parallelize_fsl_with_condor.html)

### Installation ###

1) PIP Install (recommended)
```
sudo pip install tfce-mediation
```
2) Install from source

```
git clone https://github.com/trislett/TFCE_mediation.git
cd TFCE_mediation
sudo python setup.py install
```

### Example work flow for vertex-based analyses ###

1) Run standard [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki) (for vertex-based analyses) recon-all preprocessing of T1-weighted images.

2) Spherical regrestration


```
tfce_mediation step0-vertex -i subjectslist.csv area -p
```

Explanation:
Perform a spherical surface registration for each subject to the fsaverage resampling either area or thickness.

Inputs:
* subjectslist.csv (a text file contain one column of subject IDs. Note, the IDs must match the folders those in the $SUBJECTS_DIR folder.)
* area (the surface to include in the analysis. For basic use, this should be area or thickness)

Outputs:
* ?h.all.area.00.mgh (all subjects to the fsaverage template using spherical registration)
* ?h.all.area.03B.mgh (the above file after 3mm FWHM smoothing of the surface)

2) Optional: Box-Cox transformation of the white matter surface

```
tm_tools vertex-box-cox-transform -i lh.all.area.00.mgh 8
tm_tools vertex-box-cox-transform -i rh.all.area.00.mgh 8

# replace the output
mv ?h.all.area.03B.boxcox.mgh ?h.all.area.00.boxcox.mgh
```

Explanation:
It has been suggested that surface area follows roughly a lognormal distribution [(Winkler, et al., 2012)](https://surfer.nmr.mgh.harvard.edu/ftp/articles/2012/2012_-_Winkler_et_al._-_NeuroImage.pdf); therefore, vertex-box-cox-transform normalizes the unsmoothed images using a power transformation (Box-Cox) transformation.

Inputs:
* ?h.all.area.00.mgh (The unsmoothed concatenated surface area image for subjects included)
* 8 (The number of processers to use)

Outputs:
* ?h.all.area.00.boxcox.mgh (Box-Cox transformed image)
* ?h.all.area.03B.boxcox.mgh (Box-Cox transformed image after 3mm FWHM smoothing)

3) Optional: orthonormalizing the regressors

```
tm_tools regressor-tools -i predictors.csv covariates.csv -o -s
```

Explanation:

For the two-step multiple regression and mediation analyses using TFCE_mediation, it is recommended to scale (or whiten with orthonormalization) the regressors. The input file(s) should be dummy coded, and deliminated with comma. The program returns either the orthogonalization of the input file(s) or it returns the residuals from a least squares regression to remove the effect of covariates from variable. In this example, we using the orthonormalization option (-o -s).

Inputs:
* predictors.csv (dummy-coded regressors of interest)
* covariates.csv (dummy-coded regressors of no interest)

Outputs:
* predictors_orthogonized.csv (orthogonormalized regressors of interest)
* covariates_orthogonized.csv (orthogonormalized regressors of no interest)

4) Multiple Regression

```
mv ?h.all.area.03B.boxcox.mgh ?h.all.area.00.boxcox.mgh
tfce_mediation step1-voxel-regress -i predictors_orthogonized.csv covariates_orthogonized.csv
```


