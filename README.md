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

### Example work-flow for vertex analyses ###

1) Run standard [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki) (for vertex-based analyses) recon-all preprocessing of T1-weighted images.

2) Spherical regrestration


```
tfce_mediation step0-vertex -i subjectslist.csv area -p
```

Explanation:
Perform a spherical surface registration for each subject to the fsaverage resampling either area or thickness.

Inputs:
* subjectslist.csv (a text file contain one column of subject IDs. Note, the IDs must match those in the $SUBJECTS_DIR folder.)
* area (the surface to include in the analysis. For basic use, this should be area or thickness)

Outputs:
* ?h.all.area.00.mgh (all subjects to the fsaverage template using spherical registration)
* ?h.all.area.03B.mgh (the above file after 3mm FWHM smoothing of the surface)

3) Optional: Box-Cox transformation of the white matter surface

```
tm_tools vertex-box-cox-transform -i lh.all.area.00.mgh 8
tm_tools vertex-box-cox-transform -i rh.all.area.00.mgh 8

# replace the ?h.all.area.03B.mgh with ?h.all.area.03B.boxcox.mgh
for i in lh rh; do
	mv ${i}.all.area.03B.mgh ${i}.all.area.00.backup.mgh;
	mv ${i}.all.area.03B.boxcox.mgh ${i}.all.area.03B.mgh;
done
```

Explanation:
It has been suggested that surface area follows roughly a lognormal distribution [(Winkler, et al., 2012)](https://surfer.nmr.mgh.harvard.edu/ftp/articles/2012/2012_-_Winkler_et_al._-_NeuroImage.pdf); therefore, vertex-box-cox-transform normalizes the unsmoothed images using a power transformation (Box-Cox) transformation.

Inputs:
* ?h.all.area.00.mgh (The unsmoothed concatenated surface area image for subjects included)
* 8 (The number of processers to use)

Outputs:
* ?h.all.area.00.boxcox.mgh (Box-Cox transformed image)
* ?h.all.area.03B.boxcox.mgh (Box-Cox transformed image after 3mm FWHM smoothing)

4) Optional: orthonormalizing the regressors

```
tm_tools regressor-tools -i predictors.csv covariates.csv -o -s
```

Explanation:
For the two-step multiple regression and mediation analyses using TFCE_mediation, it is recommended to scale (or orthonormalization) the regressors. The input file(s) should be dummy coded, and comma deliminated. The program returns either the orthogonalization of the input file(s) or it returns the residuals from a least squares regression to remove the effect of covariates from variable. In this example, we using the orthonormalization option (-o -s).

Inputs:
* predictors.csv (dummy-coded regressors of interest)
* covariates.csv (dummy-coded regressors of no interest)

Outputs:
* predictors_orthogonized.csv (orthogonormalized regressors of interest)
* covariates_orthogonized.csv (orthogonormalized regressors of no interest)

5) Multiple Regression

```
tfce_mediation step1-voxel-regress -i predictors_orthogonized.csv covariates_orthogonized.csv -s area
```

Explanation:
A two-step multiple regression is performed, and the resulting T-statistic images then undergo TFCE.

Inputs:
* predictors_orthogonized.csv (orthogonormalized regressors of interest)
* covariates_orthogonized.csv (orthogonormalized regressors of no interest)
* area (the surface of interest)

Outputs:
* python_temp_area/*.npy (Memory mapped numpy objects used later for permutation testing)
* output_area/tstat_area_?h_con?.mgh (t-statistic image or the respective hemispehere and contrast)
* output_area/negtstat_area_?h_con?.mgh (negative t-statistic image or the respective hemispehere and contrast)
* output_area/tstat_area_?h_con?_TFCE.mgh (TFCE transformed t-statistic image)
* output_area/negtstat_area_?h_con?_TFCE.mgh (TFCE transformed negative t-statistic image)
* output_area/max_TFCE_contrast_values.csv (The max TFCE value for the positive and negative contrast(s))

6) Permuation Testing (Randomization)

```
tfce_mediation step2-randomise-parallel --vertex area -n 10000 -p 8
```

Explanation:
Permutation testing using parallel processing. This script is a wrapper to make n=200 permution chunks, and parallelizing processing of the vertex-regress-randomise scipt for each chunk (in this case). i.e., each chunk to a different processor.

Input: 

* area (surface of interest)
* 10000 (total number of permutations)
* 8 (number of cores to used with GNU parallel)

Output:

* output/perm_Tstat_area/perm_tstat_con?_TFCE_maxVertex.csv (the maximum TFCE value among all vertices of the entire cortex for each permutation. It is used to correct for family-wise error)

7) Apply family-wise error rate correction

```
cd output
for i in lh rh; do 
	tfce_mediation vertex-calculate-fwep -i tstat_area_${i}_con1_TFCE.mgh perm_Tstat_area/perm_tstat_con1_TFCE_maxVertex.csv
	tfce_mediation vertex-calculate-fwep -i negtstat_area_${i}_con1_TFCE.mgh perm_Tstat_area/perm_tstat_con1_TFCE_maxVertex.csv
done
```
Explanation:
Calculate 1-P(FWE) vertex image from max TFCE values from randomisation.

Input:
tstat_area_?h_con?_TFCE.mgh (TFCE transformed T-statistic surface image)
perm_tstat_con?_TFCE_maxVertex.csv (List with maximum TFCE values) 
Output:
tstat_area_?h_con?_TFCE_FWEcorrP.mgh (1-P(FWE) corrected image)

8) View results

```
tm_tools vertex-freeview-quick -i tstat_area_lh_con1_TFCE_FWEcorrP.mgh tstat_area_rh_con1_TFCE_FWEcorrP.mgh

```


