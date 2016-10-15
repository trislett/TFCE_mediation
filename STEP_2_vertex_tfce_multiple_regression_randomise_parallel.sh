#! /bin/bash

#    Wrapper for parallelizing randomise multiple regression with TFCE
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
	echo ${SCRIPTPATH}/vertex_tfce_multiple_regression_randomise.py $(($i*100+1)) $(($i*100+100)) ${surf}
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
