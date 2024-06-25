#!/bin/bash
#
#set -u
set -e
#set -x 
########
while read M2L; do if [[ ${M2L:0:1} != "#" ]]; then module load $M2L; fi ; done</users_home/oda/ag15419/tobeloaded.txt
########
in_date="20191112"
cdo_lonlatbox="-19,42,30,48"
cdo_expr="W10=sqrt(U10M*U10M+V10M*V10M);U10=U10M;V10=V10M"
system_version="EAS4"
work_date="20191112"

if [[ $system_version == "EAS5" ]]; then
   in_dir="/data/oda/med_dev/mfs_mod_ECMWF_w08_orig/OUT/${in_date}/"
   work_dir="/data/oda/med_dev/mfs_mod_ECMWF_w08_orig/MOD2/${in_date}/"
   #
   infile_name=${work_date}-ECMWF---AM0125-MEDATL-b20191113_an-fv10.00
   outfile_name="${infile_name}_W10"
   echo "Working on ${in_dir}/${infile_name}.nc to ${work_dir}/${outfile_name}.nc"
   cdo -sellonlatbox,${cdo_lonlatbox} -expr,"${cdo_expr}" ${in_dir}/${infile_name}.nc ${work_dir}/${outfile_name}.nc


elif [[ $system_version == "EAS4" ]]; then
   in_dir="/data/oda/med_dev/mfs_mod_ECMWF_w08_FC_EAS4/"
   work_dir="/data/oda/med_dev/mfs_mod_ECMWF_w08_FC_EAS4/MOD2/"
   #
   in_date_plus_10=$( date -u -d "${in_date} 10 day" +%Y%m%d )
   infile_name="${in_date}_${in_date_plus_10}-ECMWF---AM0125-MEDATL-b${in_date}_fc12-fv06.00"
   outfile_name="${infile_name}_W10"
   echo "Working on ${in_dir}/${infile_name}.nc to ${work_dir}/${outfile_name}.nc"
   cdo -sellonlatbox,${cdo_lonlatbox} -expr,"${cdo_expr}" ${in_dir}/${infile_name}.nc ${work_dir}/${outfile_name}.nc
fi
##########
