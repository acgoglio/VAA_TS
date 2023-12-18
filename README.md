###
# Extraction of time-series:
# 1 -> build your p_extr.ini (ini file for extraction) and link it; you can choose to extract AN or 1st, 2nd, 3rd days of fcst (FC1,FC2,FC3) or to estract fc as a 3day time serie (FCall) otherwise you can extract MSL from atm forcings; the *long.ini correspond to the extractions for the computation of the offset (mean over 1 month of analisys)   
# 2 -> Chose the pextr job and link it (WARNING: for FCall use the same of the AN while for the forecast use FC1, FC2 and FC3 respectively for the 1st,2nd and 3rd days of fcst concatenation)
# 3 -> run runextr.sh
# 
# Fill gaps in ts with nans (modify the infos on the file names and path in the script):
# 4 -> run add_missing_ts.sh
# 
# Plot:
# 5 -> run plot*.py (plot_VAA_allTS.py for AN and FCall; plot_VAA_ANandFC123.py for AN and FC1, FC2 and FC3  )
#
# (subsample_atmforc.sh is the script that was used to create atm forcing subsampling from 3h to 6h)
# (wind.sh is the script that was used to compute the wind speed for wind maps)
# (MSLconversion.sh is the script to convert MSL Pressure from hPa to Pa)
# (max_ts.sh is the script to extract the max values)

# Plots last version of scripts
# plot_VAA_ANandFC123_osr5_new.py -> SSH time-series 2019 plot with only FC1 FC2 FC3 Obs and Tides (similar to the one in OSR5) 
# plot_VAA_atm_ANandFC123_new.py -> MSL time-series 2019 all emcwf datasets 
# plot_VAA_atm_ANandFC123_new_2022.py -> MSL time-series 2022 only FC1 FC2 FC3 Obs
# plot_VAA_wind_ANandFC123_new_2022.py -> WIND SPEED time-series 2022 all emcwf datasets
# plot_VAA_wind_ANandFC123_new.py -> WIND SPEED time-series 2019 all emcwf datasets
# plot_VAA_ANandFC123_seiches_new.py -> -> all ENSAMBLE ONLY Seiches TIME-SERIES 2019 (WARNING:non reliable results because input time-series is too short) 
# plot_VAA_ANandFC123_tides_new.py -> all ENSAMBLE ONLY TIDES TIME-SERIES 2019 (WARNING:non reliable results because input time-series is too short) 
# plot_VAA_ANandFC123_new.py -> all ENSAMBLE ONLY SSH TIME-SERIES 2019 
# plot_VAA_ANandFC123_surges_new.py -> (WARNING:non reliable results because input time-series is too short) 
# plot_VAA_winddir_ANandFC123_new.py -> WIND DIR time-series 2019 all emcwf datasets

# All other plots (Residual/Tides and Seiches) are in another repository: FFT_filter/punctual

# TMP EAS4 FC :
# 09 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191109/20191109_20191119-ECMWF---AM0125-MEDATL-b20191109_fc12-fv06.00.nc
# 10 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191110/20191110_20191120-ECMWF---AM0125-MEDATL-b20191110_fc12-fv06.00.nc
# 11 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191111/20191111_20191121-ECMWF---AM0125-MEDATL-b20191111_fc12-fv06.00.nc
# 12 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191112/20191112_20191122-ECMWF---AM0125-MEDATL-b20191112_fc12-fv06.00.nc
# 13 --> /data/oda/ec04916/DATA/FCST_AcquaAlta/Atmo/ECMWF/20191113/20191113_20191123-ECMWF---AM0125-MEDATL-b20191113_fc12-fv06.00.nc



