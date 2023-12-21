###
# Extraction of time-series:
# 1 -> build your p_extr.ini (ini file for extraction) and link it in p_extr.ini; you can choose to extract AN or 1st, 2nd, 3rd days of fcst (FC1,FC2,FC3) or to estract fc as a 3day time serie (FCall) otherwise you can extract MSL from atm forcings; the *long.ini correspond to the extractions for the computation of the offset (mean over 1 month of analisys)   
# 2 -> Chose the pextr job and link it in pextrjob_oldTG.temp (WARNING: for FCall use the same of the AN while for the forecast use FC1, FC2 and FC3 respectively for the 1st,2nd and 3rd days of fcst concatenation)
# 3 -> run runextr.sh
# 
# Fill gaps in ts with nans (modify the infos on the file names and path in the script):
# 4 -> run add_missing_ts.sh
# 
# Plot:
# 5 -> run plot*.py (plot_VAA_allTS.py for AN and FCall; plot_VAA_ANandFC123.py for AN and FC1, FC2 and FC3  )
#
# (subsample_atmforc.sh is the script that was used to create atm forcing subsampling from 3h to 6h)
# (wind.sh is the script that was used to compute the wind spead for wind maps)
# (MSLconversion.sh is the script to convert MSL Pressure from hPa to Pa)
# (max_ts.sh is the script to extract the max values)

# TMP EAS4 FC :
# 09 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191109/20191109_20191119-ECMWF---AM0125-MEDATL-b20191109_fc12-fv06.00.nc
# 10 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191110/20191110_20191120-ECMWF---AM0125-MEDATL-b20191110_fc12-fv06.00.nc
# 11 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191111/20191111_20191121-ECMWF---AM0125-MEDATL-b20191111_fc12-fv06.00.nc
# 12 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191112/20191112_20191122-ECMWF---AM0125-MEDATL-b20191112_fc12-fv06.00.nc
# 13 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191113/20191113_20191123-ECMWF---AM0125-MEDATL-b20191113_fc12-fv06.00.nc


