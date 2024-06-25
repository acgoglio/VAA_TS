#!/bin/bash
#
#set -u
set -e
#set -x 
########
while read M2L; do if [[ ${M2L:0:1} != "#" ]]; then module load $M2L; fi ; done</users_home/oda/ag15419/tobeloaded.txt
########
work_dir='/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_sea_level/'
for TOC in $( ls ${work_dir}/*.nc ) ; do 
   echo -en "$TOC max 12 Nov 20:30 21:30  " 
   cdo seldate,2019-11-12T20:30:00,2019-11-12T21:30:00 $TOC ${work_dir}/tmp.nc
   num1=$( cdo infon ${work_dir}/tmp.nc | grep "20:30" | cut -f 5 -d":" )
   num1=$( echo "$num1 + 0.7524 + 0.0995" | bc -l )
   num2=$( cdo infon ${work_dir}/tmp.nc | grep "21:30" | cut -f 5 -d":" )
   num2=$( echo "$num2 + 0.7524 + 0.0995" | bc -l )
   echo "$num1 $num2"
   rm ${work_dir}/tmp.nc
done
for TOC in $( ls ${work_dir}/*tpxo*.nc ) ; do
   echo -en "$TOC max 12 Nov 20:30 21:30  " 
   cdo seldate,2019-11-12T20:30:00,2019-11-12T21:30:00 $TOC ${work_dir}/tmp.nc
   num1=$( cdo infon ${work_dir}/tmp.nc | grep "20:30" | cut -f 5 -d":" )
   num1=$( echo "$num1 + 0.7524 + 0.0004" | bc -l )
   num2=$( cdo infon ${work_dir}/tmp.nc | grep "21:30" | cut -f 5 -d":" )
   num2=$( echo "$num2 + 0.7524 + 0.0004" | bc -l )
   echo "$num1 $num2"
   rm ${work_dir}/tmp.nc
done
for TOC in $( ls ${work_dir}/*obs*.nc ) ; do
   echo -en "$TOC max 12 Nov 20:30 21:30  " 
   cdo seldate,2019-11-12T20:30:00,2019-11-12T21:30:00 $TOC ${work_dir}/tmp.nc
   num1=$( cdo infon ${work_dir}/tmp.nc | grep "20:30" | cut -f 5 -d":" )
   num1=$( echo "$num1 + 0.7524 + 0.0004" | bc -l )
   num2=$( cdo infon ${work_dir}/tmp.nc | grep "21:30" | cut -f 5 -d":" )
   num2=$( echo "$num2 + 0.7524 + 0.0004" | bc -l )
   echo "$num1 $num2"
   rm ${work_dir}/tmp.nc
done
