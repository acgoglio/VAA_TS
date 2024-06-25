#!/bin/bash
#
#set -u
set -e
#set -x 
########
while read M2L; do if [[ ${M2L:0:1} != "#" ]]; then module load $M2L; fi ; done</users_home/oda/ag15419/tobeloaded.txt
########
in_yyyymm="201911"
system_version="eas4"
ECMWF_type="w08_12"
ECMWF_original_dir="/data/oda/med_dev/ECMWF_Venezia_AA_2019/mfs_mod_ECMWF_${ECMWF_type}_${system_version}"
ECMWF_mod_dir="${ECMWF_original_dir}_Pa_int"
ECMWF_2be_converted="${ECMWF_mod_dir}/${in_yyyymm}??/ECMWF18V10/NETCDF/${in_yyyymm}*.00.nc"
cdo_expr="MSL=MSL*100"
time_steps_perfile=8

# Copy the original dir to a new one to be modified
echo "Create the new dir: $ECMWF_mod_dir"
cp -r $ECMWF_original_dir $ECMWF_mod_dir

# Conversion of MSL Pressure field from hPa to Pa
for TOCH in $( ls $ECMWF_2be_converted ); do
    echo " Conversion MSL form hPa to Pa in file: $TOCH"
    mv $TOCH $TOCH_old.nc 
    cdo aexpr,"${cdo_expr}" $TOCH_old.nc $TOCH
    rm $TOCH_old.nc
done 

# Interpolation of all fields where time steps are missing (the first day forecasts have gaps in the first 12 hours)
echo "Check the num of time steps per file and fill the gaps interpolating.."
for TOCH in $( ls $ECMWF_2be_converted ); do
    TS_IN_FILE=$( ncdump -h $TOCH | grep "UNLIMITED" | cut -f 2 -d"(" | cut -f 1 -d "c" )
    TS_IN_FILE=${TS_IN_FILE:0:1}
    FILE_INI_DATE=$( cdo infon $TOCH | head -n 2 | tail -n 1 | cut -f 2 -d":" | cut -f 2 -d" " )
    echo "File $TOCH ,for day $FILE_INI_DATE , has $TS_IN_FILE of $time_steps_perfile time-steps per file"
    if [[ $TS_IN_FILE -lt $time_steps_perfile ]]; then
       echo "Interpolation is needed.."
       mv $TOCH $TOCH_old.nc
       cdo inttime,${FILE_INI_DATE},00:00:00,3hour $TOCH_old.nc $TOCH
    fi
done

##########
