# Surf_tfce #

This is a placeholder for the full instructions.

### What is this repository for? ###

* Quick summary
* Version

### Dependencies ###

Surf_tfce has been tested on Ubuntu 14.04 and 16.04, Arch Linux, OSX

* [SciPy Stack](https://www.scipy.org/install.html)
* [NiBabel](http://nipy.org/nibabel/installation.html#installation)
* [Cython](http://cython.org)
* [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki)

Recommended

* [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki)
* [Neurodebian reprository](http://neuro.debian.net/)

Recommended for parallelization

* [GNU Parallel](http://www.gnu.org/software/parallel/)

```
#!bash

sudo apt-get install parallel
```

* For personal installation of [HTCondor](https://research.cs.wisc.edu/htcondor/) (i.e., Condor), you can follow the instructions [here](http://neuro.debian.net/blog/2012/2012-03-09_parallelize_fsl_with_condor.html)

1) Install the required python dependences. If you are using ubuntu or debian run:
```
#!Bash


sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose cython python-nibabel
```
2) Download this repository to somewhere useful (e.g., scripts directory):

```
#!Bash
git clone https://bitbucket.org/trislett/surf_tfce.git
```
3) It is recommended to recompile the cython and c++ scripts. 

Optional:    
bash_compile.sh '-e' option includes a link in your .bashrc to the script directory as an environment variable $SURF_TFCE.   
bash_compile.sh '-m' option copies ?h.midthickness surface to fsaverage/surf/ directory. Depending on how you installed freesurfer you may have to run 'sudo ./bash_compile.sh -e -m'    

```
#!Bash
cd src
./bash_compile.sh
```