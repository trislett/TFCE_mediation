# TFCE_mediation
Fast regression and mediation analysis of vertex or voxel MRI data with TFCE

![Splash type schematic](tfce_mediation/doc/Mediation_artboard.png "Schematic")

### Citation ###

[Lett TA, Waller L, Tost H, Veer IM, Nazeri A, Erk S, Brandl EJ, Charlet K, Beck A, Vollstädt-Klein S, Jorde A, Keifer F, Heinz A, Meyer-Lindenberg A, Chakravarty MM, Walter H. Cortical Surface-Based Threshold-Free Cluster Enhancement and Cortexwise Mediation. Hum Brain Mapp. 2017 March 20. DOI: 10.1002/hbm.23563](http://onlinelibrary.wiley.com/doi/10.1002/hbm.23563/full)

The pre-print manuscript is available [here](tfce_mediation/doc/Lett_et_al_2017_HBM_Accepted.pdf) as well as the [supporting information](tfce_mediation/doc/Lett_et_al_2017_HBM_supporting_information.docx).

### Installation ###
See wiki [Install TFCE_mediation](https://github.com/trislett/TFCE_mediation/wiki/Install-TFCE_mediation)

### Tutorials ###

See wikis:

[Work-flow example: Vertex-wise regression analyses](https://github.com/trislett/TFCE_mediation/wiki/Work-flow-example-for-vertex-wise-regression-analyses)

[Work-flow example: Voxel-wise mediation analyses](https://github.com/trislett/TFCE_mediation/wiki/Work-flow-example-for-voxel-wise-mediation-analyses)

[Work-flow example: Multimodal multisurface regression (mmr)](https://github.com/trislett/TFCE_mediation/wiki/Work-flow-example-for-multimodal,-multisurface-regression-(MMR))

Additional help:
* Every command and subcommand has an extensive help function (i.e., try -h)
* Ask in the [Issues](https://github.com/trislett/TFCE_mediation/issues) section, even if it is just a question.

### What's new / updates ###
17-11-2018
* version 1.6.0
* An alpha version of step1-vertex-mixed is now available
* Features included in step1-vertex-mixed: (1) a new GLM implementation, (2) one factor repeated measure ANCOVA, (3) two factor repeated measure ANCOVA.
* step1-vertex-mixed also has a new input interface where just a csv file needs to be read and the variable will be dummy coded automatically.
* Support for voxel based analyses as well as mmr/tmi will be released shortly.

8-02-2018
* version 1.5.0 is now availale on [pypipe (PIP)](https://pypi.org/project/tfce-mediation/).
* TFCE_mediation now supports python 2.7 and python 3.5.
* Changes were made to TMI format I/O to accommodate python 3.

29-01-2018
* version 1.4.2 is now available on [pypipe (PIP)](https://pypi.org/project/tfce-mediation/).
* Small bug fixes. Added small more human readable error messages.
* ~~Updating to numpy 1.14.0 will give a warning about the package h5py (used to import minc images). It can be ignored, or removed by downgraging to numpy 1.13.3 (sudo -H pip install numpy==1.13.3).~~ (h5py is no longer required).

22-12-2017

* version 1.4.0 is now available on [pypipe (PIP)](https://pypi.org/project/tfce-mediation/).
* 1.4 includes numerous updates to tm_multimodality include the addition of mmr-lr (multimodality, multisurface regression - low RAM) which vastly reduces the RAM requires for joint multimodal analyses of TMI files. For example, it ran 10000 permutations of a 10.5 GB TMI file contains 50 surfaces overnight (~ 7 million vertices by 350 subjects) using approximately 25 GB of RAM. After the permutation testing has completed, study-wide 1-p(FWER) can be created using the standard mmr technique (i.e., corrected for the maximum TFCE value among ~7 million vertices per permutation).
* mmr-lr supports mixed TFCE setting, so it is possible to include voxelwise images (e.g., TBSS skeleton) with vertexwise images (cortical thickness, etc.) in the same analysis and correcting across all modalities.
* mmr-lr, like mmr, supports mediation modeling.
* Added a WIKI for mmr/mmr-lr.

26-10-2017

* version 1.3.4-1.3.5: small bug fixes, and various alpha features for tm_multimodality.
* version 1.3.6 is now available on [pypipe (PIP)](https://pypi.org/project/tfce-mediation/).
* Added an option to include vertex or voxel 4D images as covariates to multiple-regression analyses. Variance Inflation Factor (VIF) images are automatically outputted to check for multicollinearity among independent variables (only an issue if the VIF is approximately > 5). 

30-8-2017

* version 1.3.3 is now available on [pypipe (PIP)](https://pypi.org/project/tfce-mediation/).
* tm_multimodality is now available which includes: 
	* mmr (multimodality, multisurface regression)
	* mmr-parallel (parallel mmr for permutation testing)
	* read-tmi-header (convenient reader for the TMI ascii header)
	* create-tmi (create TMI files using almost any neuroimaging filetype and surface file)
	* edit-tmi (manipulate TMI file using its history, reorder tmi elements, and much more)
	* multimodal-adjacency (create adjacency set for almost any type of neuroimaging data)
* The most unique feature of tm_multimodality is the ability to perform FWER correction across all image modalities with mixed TFCE settings. Although this feature has been validated (email me for the details), tm_multimodality is currently an alpha build.
* Added a wiki for the [TMI file format](https://github.com/trislett/TFCE_mediation/wiki/TMI-Neuroimaging-File-Format)
* Integrated tfce_mediation with the, just released, [TMI_viewer](https://github.com/trislett/tmi_viewer)! TMI_viewer is a standalone viewer for TMI files. TFCE_mediation tm_multimodality could already export to PLY, MGH, or NIFTI files. However, tmi_viewer directly renders any voxel- or vertex-based image contained in a TMI file. [Example: TBSS voxel skeleton and Midthickness SA](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/doc/tmi_viewer_multimodal.png)

19-07-2017

* version 1.2.2 is now available on [pypipe (PIP)](https://pypi.org/project/tfce-mediation/).
* fixes a bug in the permutation testing of surfaces with unequal sizes

5-07-2017

* version 1.2.1 is now available on [pypipe (PIP)](https://pypi.org/project/tfce-mediation/). 
* A number of small fixes have been made. 
* Release of an alpha version of multi-surface, multi-modality regression (mmr) analysis using the tmi image. The scripts can be found [here](https://github.com/trislett/TFCE_mediation/tree/master/tfce_mediation/tm_multisurface).
* [tm_multimodality_multisurface_regression.py](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/tm_multisurface/tm_multimodality_multisurface_regression.py) is still in the early stages but it already has many new features including:
	- In the same set of subjects, cortical thickness, surface area, neurite density, and fMRI contrast maps could be packaged in the a tmi file using [create_tmi.py](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/tools/create_tmi.py). Statistical analysis (and randomization) across all modalities can be performed using [tm_multimodality_multisurface_regression.py](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/tm_multisurface/tm_multimodality_multisurface_regression.py)
	- Study-wide FWE correction with TFCE. i.e., 1-p(FWE<0.05) images are produced that are corrected via a standardization algorithm and maximum TFCE values among all surfaces.
	- The results can be exported as ply files and viewed using Blender or meshlab. Alternatively, the tmi result files can be converted to mgh or nifti. 
	- Many surfaces can be analyzed at once. For example, analysis among all subcortical structures can be performed with TFCE and corrected for all surfaces (or even subcortical, cortical surfaces can be combined together in a single analysis). 
* Due to the flexibility of the tmi image format, many more features will be released in the new feature. For instance, the tmi format can include both voxel and vertex data in the same space. This allows voxel-based (such as TBSS) analyses and vertex-based (such cortical thickness) to be performed concurrently with TFCE and FWE correction among all data analyzed.
* TLDR, why do volumetric analyses when surface area and cortical thickness can be analyzed as the same time? Bonferroni correction among different imaging modalities is not necessary using [tm_multimodality_multisurface_regression.py](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/tm_multisurface/tm_multimodality_multisurface_regression.py) and the [tmi image format](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/tm_io.py).

21-06-2017

* version 1.2 is now available on [pypipe (PIP)](https://pypi.python.org/pypi/tfce-mediation/1.2.0)
* added the ability to read and write binarized PLY files using 'tm_tools convert-surface'
* a preview of the new tfce_mediation imaging (tmi) for format group analyses will be released before OHBM 2017. It is based on the the PLY file format. Some advantages include a fast, space efficient, and expandable neuroimaging that harmonizes voxel and vertex images as well as surfaces all with a file history function, but more on that later. 

16-05-2017

* added an option to use geodesic FWHM smoothing in tfce_mediation step0-vertex. Geodesic smoothing requires a distances list that can either be created using [fwhm_compute_distances_parallel.py](https://github.com/trislett/TFCE_mediation/tree/master/tfce_mediation/misc_scripts) or downloaded from [tm_addons](https://github.com/trislett/tm_addons). Currently, only the midthickness surface is supported, but other surfaces will be supported soon. 

13-05-2017

* convert-surface now supports painting freesurfer label files to .ply files as well as outputing a legend. An example of the lh aparc atlas on the midthickness surface is: [left view](https://github.com/trislett/tm_addons/blob/master/4kRender/lh.aparc.annot_left_1080p.png), [right view](https://github.com/trislett/tm_addons/blob/master/4kRender/lh.aparc.annot_right_1080p.png), [legend](https://github.com/trislett/tm_addons/blob/master/4kRender/lh.aparc.annot_legend.png), and the lh [Glasser et al. (2016)](http://www.nature.com/nature/journal/v536/n7615/abs/nature18933.html) HCP-MMC1 atlas: [left view](https://github.com/trislett/tm_addons/blob/master/4kRender/lh.HCP-MMP1.annot_left_1080p.png), [right view](https://github.com/trislett/tm_addons/blob/master/4kRender/lh.HCP-MMP1.annot_right_1080p.png), and a ridiculously long [legend](https://github.com/trislett/tm_addons/blob/master/4kRender/lh.HCP-MMP1.annot_legend.png)
* Support for loading 4D minc volumes with tfce_mediation step0-voxel. Right now the outputs will still be nifti (which could be converted to mnc e.g. by using nii2mnc).

05-05-2017

* convert-surface now supports exporting the [Polygon File Format (PLY)](https://en.wikipedia.org/wiki/PLY_(file_format)). Statistic files (*.mgh) can now be painted on the vertices with specifying a threshold range (e.g 0.95 1 for pFWE corrected images) with an either red-yellow or blue-lightblue color scheme (as well as all [matplotlib color schemes](https://matplotlib.org/examples/color/colormaps_reference.html)). So now you can 3D print your surface based results!
* A few examples of the high quality (3840x2160) render of painted results of the ?h.mean.area.00.mgh from 0.1mm to 1.0mm using the 'jet' colormap [right](https://github.com/trislett/tm_addons/blob/master/4kRender/MeanArea_transparent_right.png), [rostral](https://github.com/trislett/tm_addons/blob/master/4kRender/MeanArea_transparent_rostral.png), and [superior](https://github.com/trislett/tm_addons/blob/master/4kRender/MeanArea_transparent_superior.png). The surfaces were rendered using [Blender](https://www.blender.org/) and exported *.ply objects.
* A sample blender scene for making figures can be downloaded from [here](https://github.com/trislett/tm_addons/blob/master/BlenderScene/Sample_Scene.blend). n.b., when importing a new *ply object, the scale will need to be reduced. 
* wiki is slowly becoming alive

03-05-2017

* Added surface conversion tool (tm_tools convert-surface) for converting gifti / MNI objects / freesurfer .srf files to freesurfer .srf files for analysis with TFCE_mediation or waveform objects / Stl objects for importing the surface to 3D rendering software such as Blender or Meshlab. Among other things, this means that there is now full support for statistical analysis of neuroimages processed with CIVET. Here is an example of the midthickness surface [LH](https://github.com/trislett/tm_addons/blob/master/3dSurfaces/lh.midthickness.stl) and [RH](https://github.com/trislett/tm_addons/blob/master/3dSurfaces/rh.midthickness.stl).

02-05-2017

* Added fwhm_compute_distances_parallel to the misc_scripts directory. It is necessary for geodesic FWHM smoothing. 
* Full support for CIFTI-2 and GIFTI in the very near future.
* Version 1.1 is now available on pip 

21-04-2017

* Geodesic FWHM smoothing on the vertex images. It is performed on the midthickness surface. The runtime is approximately 4 minutes per image, but it can probably be faster. No more fudge factors!
* FWHM smoothing on voxel images.
* Too many additions to tm_maths than can be listed. Some examples are surface-based ICA artefact removal, curve fitting for calculation of the max TFCE null distribution of FWE correction, pca compression smoothing, many clustering algorithms, and powerful normalization tools. Check the help function.
* Initial support for CIVET vertex-based analyses (see the issues section)
* Version 1.1 will be pushed to pip in the near future. 
