#! /bin/bash

if [ $# -eq 0 ]; then
	echo "Wrapper for parallelizing randomise"
	echo "Usage: `basename $0` [-s <(area or thickness)>] [-n <# of permutations>] [-p <# of processors> OR -c OR -f]"
	echo "For GNU Parallel: -p [# of processors]"
	echo "For Condor: -c"
	echo "For fsl_sub: -f"
	exit 1;
fi


SCRIPT=$0
SCRIPTPATH=`dirname $SCRIPT`
current_time=$(date +%Y%m%d%H%M%S)

while getopts "s:n:p:cf" opt; do
	case $opt in
		s)
      			surf=$OPTARG
		;;
		n)
      			numberperm=$OPTARG
			outperm=`expr '(' $numberperm + 50 ')'  / 100 '*' 100 '/' 2`
			forperm=$((($outperm/100)-1))
		;;
		p)
			num_processors=$OPTARG
			p_opt="gnu"
      		;;
		c)
			p_opt="condor"
		;;
		f)
			p_opt="fsl_sub"
		;;
    		\?)
      			echo "Invalid option: -$OPTARG"
			exit 1
      		;;
	esac
done

roundnumperm=$(($forperm*2*100+200))
echo "Evaluating $roundnumperm permuations"
for i in $(eval echo "{0..$forperm}"); do 
	echo ${SCRIPTPATH}/surf_tfce_multiple_regression_vertexTFCE_randomise.py $(($i*100+1)) $(($i*100+100)) ${surf}
done > cmd_multipleregress_randomise_${surf}_${current_time}

if [[ $p_opt = "gnu" ]]; then
	cat cmd_multipleregress_randomise_${surf}_${current_time} | parallel -j ${num_processors}
fi

if [[ $p_opt = "condor" ]]; then
	${SCRIPTPATH}/tools/submit_condor_jobs_file cmd_multipleregress_randomise_${surf}_${current_time}
fi

if [[ $p_opt = "fsl_sub" ]]; then
	fsl_sub -t cmd_multipleregress_randomise_${surf}_${current_time}
fi
echo "Run: ${SCRIPTPATH}/tools/calculate_fweP_vertex.py to calculate (1-P[FWE]) image (after randomisation is finished)."
