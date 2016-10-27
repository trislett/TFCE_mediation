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

### Example vertex-based analysiss ###

1) Run standard [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki) (for vertex-based analyses) recon-all preprocessing of T1-weighted images.

2) Spherical regrestration


```
tfce_mediation step0-vertex -i sujectslist.csv area -p
```

Inputs:
	subjectslist.csv (a text file contain one column of subject IDs. Note, the IDs must match the folders those in the $SUBJECTS_DIR folder.)
	area (the surface to include in the analysis. For basic use, this should be area or thickness)

Outputs:
	?h.all.area.00.mgh (all subjects to the fsaverage template using spherical registration)
	?h.all.area.03B.mgh (the above file after 3mm FWHM smoothing of the surface)

2) Multiple regression

