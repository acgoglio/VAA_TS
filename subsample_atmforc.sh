#!/bin/bash
#
#set -u
set -e
#set -x 
########
while read M2L; do if [[ ${M2L:0:1} != "#" ]]; then module load $M2L; fi ; done</users_home/oda/ag15419/tobeloaded.txt
########
work_dir_file="/data/oda/med_dev/mfs_mod_ECMWF_w08sub_eas?/2019????/ECMWF18V10/NETCDF/*.nc"
for TOC in $( ls $work_dir_file ) ; do 
     echo "Working on file: $TOC .."
     mv $TOC ${TOC}_original.nc
     cdo seltimestep,1,3,5,7 ${TOC}_original.nc $TOC
     echo ".. Done!"
done
