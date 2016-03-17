#! /bin/bash

if [ $# -eq 0 ]; then
	echo "--- Basic wrapper freesurfer's freeview ---"
	echo "Usage: `basename $0` -i [image.mgh] -h [hemi (lh or rh)]"
	echo "   Defaults: surface=inflated, thresholds=0.95,0.985"
	echo "Options:"
	echo "-s [surface] fsaverage surface (e.g., pial, inflated sphere)"
	echo "-o [surface_file] other surface file (e.g. lh.midthickness)"
	echo "-l lower threshold"
	echo "-u upper threshold"
	echo ""
	echo "e.g., $0 -i tstat_con1_thickness_lh_TFCE_FWEcorrP.mgh -h lh -s pial"
	exit 1;
fi

dirlocation=`pwd`
fsaverge_surf=inflated
lower=0.95
upper=0.985

while getopts "i:h:s:o:l:u:" opt; do
	case $opt in
		i)
			surfname=$OPTARG
		;;
		h)
			hemi=$OPTARG
		;;
		s)
			fsaverge_surf=$OPTARG
		;;
		o)
			surf_file=$OPTARG
			use_other=1
		;;
		l)
			lower=$OPTARG
		;;
		u)
			upper=$OPTARG
		;;
		\?)
			echo "Invalid option: -$OPTARG"
			exit 1
		;;
	esac
done

if [[ $use_other = 1 ]]; then
	surf_file=${dirlocation}/${surf_file}
else
	surf_file=$SUBJECTS_DIR/fsaverage/surf/${hemi}.${fsaverge_surf}
fi

freeview -f ${surf_file}:overlay=${dirlocation}/${surfname}:overlay_threshold=${lower},${upper} -viewport 3d
