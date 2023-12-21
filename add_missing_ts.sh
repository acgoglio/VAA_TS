#!/bin/bash
#
#set -u
set -e
#set -x 
########
while read M2L; do if [[ ${M2L:0:1} != "#" ]]; then module load $M2L; fi ; done</users_home/oda/ag15419/tobeloaded.txt
########
work_dir='/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_w08_w08_12_w10_ANFC/'
in_dir='/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_w08_w08_12_w10_ANFC/'
ref_file='/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_w08_w08_12_w10_ANFC/ISMAR_TG_mod_EAS6_AN_w08.nc'
for TOC in $( ls ${in_dir}/*_mod_EAS?_*_w08_12.nc ) ; do 
   echo -en "$TOC -> " 
   NUM_OF_TS=$( ncdump -h $TOC | grep "UNLIM" | cut -f 2 -d"(" | cut -f 1 -d" " )
   echo $NUM_OF_TS
   if [[ $NUM_OF_TS != 216 ]]; then
      echo "LESS then 216"
      # Loop on all the dates 
      DATE_IDX=20191109
      while [[ $DATE_IDX -le 20191117 ]]; do
        # Check if there is a record for this date
        HOW_MANY=$( cdo infon $TOC | grep -c "${DATE_IDX:0:4}-${DATE_IDX:4:2}-${DATE_IDX:6:2}" | cut -f 1 -d" " )
        echo "Found $HOW_MANY / 24 "
        # if not build the nan record
        if [[ $HOW_MANY -lt 24 ]]; then
           echo "${DATE_IDX} missing $HOW_MANY!"
           cdo seltimestep,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 ${ref_file} ${work_dir}/tmp_ini.nc
           cdo setdate,${DATE_IDX:0:4}-${DATE_IDX:4:2}-${DATE_IDX:6:2} ${work_dir}/tmp_ini.nc ${work_dir}/tmp_${DATE_IDX}_nonnan.nc
           cdo setrtomiss,-200,200 ${work_dir}/tmp_${DATE_IDX}_nonnan.nc ${work_dir}/tmp_${DATE_IDX}_0.nc
           cdo setctomiss,0 ${work_dir}/tmp_${DATE_IDX}_0.nc ${work_dir}/tmp_${DATE_IDX}.nc
           rm ${work_dir}/tmp_ini.nc ${work_dir}/tmp_${DATE_IDX}_nonnan.nc ${work_dir}/tmp_${DATE_IDX}_0.nc
        fi
  
      DATE_IDX=$( date -u -d "${DATE_IDX} 1 day" +%Y%m%d )
      done 

      # Mergetime all the tmp files
      cdo mergetime ${work_dir}/tmp_*.nc ${TOC} ${TOC}_ok.nc
      rm ${work_dir}/tmp_*.nc
   fi 
done
