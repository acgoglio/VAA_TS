import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')

# === Input directory ===
input_dir = '/work/cmcc/ag15419/tmp_med_dev_old/Venezia_Acqua_Alta_2019/VAA_sea_level_paper/'

# === Time window ===
start_time = datetime(2019, 11, 10)
end_time   = datetime(2019, 11, 13)

# === Read NetCDF files ===
def read_nc_series(filename, varname, timevar='time_counter'):
    ds = xr.open_dataset(os.path.join(input_dir, filename))
    time = ds[timevar].values
    time = pd.to_datetime(time)
    data = ds[varname].squeeze().values * 100  # convert to cm
    return time, data

# === Read CSV OBS files ===
def read_csv_series_hourly(filename):
    df = pd.read_csv(os.path.join(input_dir, filename), header=None, sep=';', comment='#')
    time = pd.to_datetime(df[0]) - pd.Timedelta(hours=1)  # shift 1h
    data = df[1].values * 100
    return time, data

def read_csv_series_10min(filename):
    df = pd.read_csv(
        os.path.join(input_dir, filename),
        header=None,
        sep=';',
        comment='#',
        na_values=['nan']  # find nans
    )
    time = pd.to_datetime(df[0], dayfirst=True, errors='coerce')  # `errors='coerce'` trasforma date non valide in NaT
    time = time - pd.Timedelta(hours=1)  # shift 1h
    data = df[1].astype(float) * 100  # mantieni i NaN e converti a cm
    return time, data

# === Read data ===
# Mod FC_w10_20191110
time_bt, bt = read_nc_series('ISMAR_TG_mod_EAS9BT_FC_w10.nc', 'sossheig')
time_bc, bc = read_nc_series('ISMAR_TG_mod_EAS6_FCall_20191110_w10.nc', 'sossheig')
time_bc9, bc9 = read_nc_series('ISMAR_TG_mod_EAS9BC_FC_w10.nc', 'sossheig')

# Mod FC_w10_20191111-FC_w10_20191115
time_bc9_20191111, bc9_20191111 = read_nc_series('ISMAR_TG_mod_EAS9_FC_w10_20191111.nc', 'sossheig')
time_bc9_20191112, bc9_20191112 = read_nc_series('ISMAR_TG_mod_EAS9_FC_w10_20191112.nc', 'sossheig')
time_bc9_20191113, bc9_20191113 = read_nc_series('ISMAR_TG_mod_EAS9_FC_w10_20191113.nc', 'sossheig')
time_bc9_20191114, bc9_20191114 = read_nc_series('ISMAR_TG_mod_EAS9_FC_w10_20191114.nc', 'sossheig')
time_bc9_20191115, bc9_20191115 = read_nc_series('ISMAR_TG_mod_EAS9_FC_w10_20191115.nc', 'sossheig')

# Mod AN_w10
time_an6,an6 = read_nc_series('ISMAR_TG_mod_eas6_an.nc','zos','time')
time_an9,an9 = read_nc_series('ISMAR_TG_mod_EAS9_AN_w10.nc','sossheig','time_counter')
time_an9nt,an9nt = read_nc_series('ISMAR_TG_mod_EAS9-NT_AN_w10.nc','sossheig','time_counter')


# TPXO
time_tpxo, tpxo = read_nc_series('ISMAR_TG_tpxo.nc', 'tide_z', timevar='time') 

# Obs
time_obs_hourly, obs_hourly = read_csv_series_hourly('ISMAR_TG_obs_long2m.csv') #('ISMAR_TG_obs_long.csv') #('ISMAR_TG_obs_10-12.csv') 
time_obs_hf, obs_hf = read_csv_series_10min('ISMAR_TG_obs_10min.csv_ok.csv')

# Build hourly time series from 10 min freq time series
obs_hf_series = pd.Series(obs_hf, index=pd.to_datetime(time_obs_hf))
#obs_hourly_centered = obs_hf_series.resample('60min', offset='30min').mean()
#time_obs_hourly_centered = obs_hourly_centered.index.to_pydatetime()
#obs_hourly_centered_values = obs_hourly_centered.values
#time_obs_hourly, obs_hourly =time_obs_hourly_centered,obs_hourly_centered_values

# === Time filter ===
def filter_time(t, x, ini_t, end_t):
    mask = (t >= ini_t) & (t <= end_t)
    return t[mask], x[mask]


# === Plot 0a: only obs and obs avg ===

# Obs avgs (m->cm)
obs_clim_1983_2020 = 0.37
time_nov_2019, obs_Nov_2019                 = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,1), datetime(2019,12,1))
time_oct_2019, obs_Oct_2019                 = filter_time(time_obs_hourly, obs_hourly, datetime(2019,10,1), datetime(2019,11,1))
time_oct10_nov10_2019, obs_10Oct_10Nov_2019 = filter_time(time_obs_hourly, obs_hourly, datetime(2019,10,10), datetime(2019,11,11))
time_nov_1_10, obs_Nov_1_10                 = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,1), datetime(2019,11,10))
time_nov_4_10, obs_Nov_4_10                 = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,4), datetime(2019,11,10))
time_nov_10_12, obs_Nov_10_12               = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,10), datetime(2019,11,13))
time_nov_10_125, obs_Nov_10_125             = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,10,12,0), datetime(2019,11,13,12,0))
time_nov_10_13, obs_Nov_10_13               = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,10), datetime(2019,11,14))
time_nov_10_15, obs_Nov_10_15               = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,10), datetime(2019,11,16))
time_nov_10_17, obs_Nov_10_17               = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,10), datetime(2019,11,18))
time_nov_11_17, obs_Nov_11_17               = filter_time(time_obs_hourly, obs_hourly, datetime(2019,11,11), datetime(2019,11,18))

obs_clim_1983_2020   = obs_clim_1983_2020*100
obs_Nov_2019         = np.nanmean(obs_Nov_2019)
obs_Oct_2019         = np.nanmean(obs_Oct_2019)
obs_10Oct_10Nov_2019 = np.nanmean(obs_10Oct_10Nov_2019)
obs_Nov_1_10         = np.nanmean(obs_Nov_1_10)
obs_Nov_4_10         = np.nanmean(obs_Nov_4_10)
obs_Nov_10_12        = np.nanmean(obs_Nov_10_12)
obs_Nov_10_125       = np.nanmean(obs_Nov_10_125)
obs_Nov_10_13        = np.nanmean(obs_Nov_10_13)
obs_Nov_10_15        = np.nanmean(obs_Nov_10_15)
obs_Nov_10_17        = np.nanmean(obs_Nov_10_17)
obs_Nov_11_17        = np.nanmean(obs_Nov_11_17)

peak_obs_hourly = np.nanmax(obs_hourly)
peak_obs_hf = np.nanmax(obs_hf)

fig, ax = plt.subplots(figsize=(12, 8))
plt.rc('font', size=20)

# Time series
#ax.plot(time_obs_hf, obs_hf, 'b--', linewidth=2, label=f'10 min freq. OBS (12 November peak = {int(peak_obs_hf)} cm)')
ax.plot(time_obs_hourly, obs_hourly, 'o-', color='navy', label=f'Hourly OBS (12 November peak = {int(peak_obs_hourly)} cm)', markersize=4)
#ax.plot(time_tpxo, tpxo, 'k--', label='Tides TPXO9')

# Colormap Blues
y_labels = ['Obs clim 1983-2020:', 'Obs 10 Oct-10 Nov 2019 avg:', 'Obs Nov 2019 avg:', 'Obs Nov 1-10 avg:', 'Obs Nov 10-12 avg:', 'Obs Nov 10 12:00 - 13 12:00 avg:', 'Obs Nov 10-13 avg:', 'Obs Nov 10-15 avg:', 'Obs Nov 10-17 avg:']
y_values = [obs_clim_1983_2020, obs_10Oct_10Nov_2019, obs_Nov_2019, obs_Nov_1_10, obs_Nov_10_12, obs_Nov_10_125, obs_Nov_10_13, obs_Nov_10_15, obs_Nov_10_17]
cmap = plt.cm.get_cmap('cool', len(y_values))

# Plot the avg values
ax.set_xlim(datetime(2019,11,10), datetime(2019,11,13))
for i, y in enumerate(y_values):
    y_val = float(np.squeeze(y))
    ax.axhline(y_val, color=cmap(i), linewidth=2.5, label=y_labels[i]+' '+str(round(y_values[i],1))+' cm')
    #ax.text(ax.get_xlim()[0], y + 0.01, f'Linea {i+1}', color=cmap(i), fontsize=9)

for i, y in enumerate(y_values):
    ax.axhline(y, color=cmap(i), linewidth=2.5)

# Highlight the 12 novembre 2019 AA event
gray_start = datetime(2019, 11, 12, 18)
gray_end = datetime(2019, 11, 12, 22)
ax.axvspan(gray_start, gray_end, color='grey', alpha=0.3)

# Layout e legenda
ax.set_ylim(0, 200)
ax.set_ylabel('Sea Level - Sea Level Avg [cm]', fontsize=20)
ax.set_xlabel('Days of November 2019', fontsize=20)
ax.set_title('Sea Level observations at ISMAR_TG', fontsize=20)

ax.grid(True)
ax.legend(loc='upper left', fontsize=18)

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax.get_xticklabels(which='major'), fontsize=16, weight='bold')
plt.setp(ax.get_xticklabels(which='minor'), fontsize=16, color='gray')
plt.setp(ax.get_yticklabels(), fontsize=16)
ax.grid(which='major', axis='both', linestyle='-', linewidth=0.8, color='black')     # ogni giorno
ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='gray')     # ogni 6 ore
ax.tick_params(axis='x', which='minor', length=4)
ax.grid(True, which='both')

# Save the plot
plt.tight_layout()
plt.savefig('sea_level_ISMAR_TG_obs.png', dpi=300)

# === Plot 0b: only obs and obs avg on a longer period + analysis  ===

fig, ax = plt.subplots(figsize=(16, 8))
plt.rc('font', size=16)

# Colormap Blues
y_labels = ['Obs clim 1983-2020:', 'Obs Oct 2019 avg:', 'Obs Nov 2019 avg:', 'Obs Nov 10-17 avg:']
y_values = [obs_clim_1983_2020, obs_Oct_2019, obs_Nov_2019, obs_Nov_10_17]
cmap = plt.cm.get_cmap('cool', len(y_values))

# Plot the avg values
ax.set_xlim(datetime(2019,10,1), datetime(2019,12,1))
for i, y in enumerate(y_values):
    y_val = float(np.squeeze(y))
    ax.axhline(y_val, color=cmap(i), linewidth=2.5, label=y_labels[i]+' '+str(round(y_values[i],1))+' cm')

# Time series
#ax.plot(time_obs_hf, obs_hf, 'b--', linewidth=2, label=f'10 min freq. OBS (12 November peak = {int(peak_obs_hf)} cm)')
ax.plot(time_obs_hourly, obs_hourly, 'o-', color='navy', label=f'Hourly OBS (12 November peak = {int(peak_obs_hourly)} cm)', markersize=4)
#ax.plot(time_tpxo, tpxo, 'k--', label='Tides TPXO9')

# MedFS Analysis
#time_an6, an6 = filter_time(time_an6, an6, datetime(2019,10,1), datetime(2019,12,1))
#peak_an6 = np.nanmax(an6)
#ax.plot(time_an6, an6, color='olive', label=f'EAS6 MedFS analysis (12 November peak = {int(peak_an6)} cm)')

# Highlight the 12 novembre 2019 AA event
gray_start = datetime(2019, 11, 12, 18)
gray_end = datetime(2019, 11, 12, 22)
ax.axvspan(gray_start, gray_end, color='grey', alpha=0.3)

# Layout e legenda
ax.set_ylim(-200, 200)
ax.set_ylabel('Sea Level - Sea Level Avg [cm]', fontsize=20)
ax.set_xlabel('Days of October-November 2019', fontsize=20)
ax.set_title('Sea Level observations at ISMAR_TG', fontsize=20)

ax.grid(True)
ax.legend(loc='upper left', fontsize=14)

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
#ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
#ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax.get_xticklabels(which='major'), fontsize=10, weight='bold', rotation=45)
#plt.setp(ax.get_xticklabels(which='minor'), fontsize=16, color='gray')
plt.setp(ax.get_yticklabels(), fontsize=16)
ax.grid(which='major', axis='both', linestyle='-', linewidth=0.8, color='black')     # ogni giorno
#ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='gray')     # ogni 6 ore
ax.tick_params(axis='x', which='minor', length=4)
ax.grid(True) #, which='both')

# Save the plot
plt.tight_layout()
plt.savefig('sea_level_ISMAR_TG_obs_long.png', dpi=300)

#######################################

# Cut the period: start_time-end_time
time_bt, bt = filter_time(time_bt, bt, start_time, end_time)
time_bc, bc = filter_time(time_bc, bc, start_time, end_time)
time_bc9, bc9 = filter_time(time_bc9, bc9, start_time, end_time)
time_bc9_20191111, bc9_20191111 = filter_time(time_bc9_20191111, bc9_20191111, datetime(2019,11,11), datetime(2019,11,14))
time_bc9_20191112, bc9_20191112 = filter_time(time_bc9_20191112, bc9_20191112, datetime(2019,11,12), datetime(2019,11,15))
time_bc9_20191113, bc9_20191113 = filter_time(time_bc9_20191113, bc9_20191113, datetime(2019,11,13), datetime(2019,11,16))
time_bc9_20191114, bc9_20191114 = filter_time(time_bc9_20191114, bc9_20191114, datetime(2019,11,14), datetime(2019,11,17))
time_bc9_20191115, bc9_20191115 = filter_time(time_bc9_20191115, bc9_20191115, datetime(2019,11,15), datetime(2019,11,18))
time_tpxo, tpxo = filter_time(time_tpxo, tpxo, start_time, end_time)
time_obs_hourly, obs_hourly = filter_time(time_obs_hourly, obs_hourly, start_time, end_time)
time_obs_hf, obs_hf = filter_time(time_obs_hf, obs_hf, start_time, end_time)
time_an9, an9 = filter_time(time_an9, an9, start_time, end_time)
time_an9nt, an9nt = filter_time(time_an9nt, an9nt, start_time, end_time)

# === Plot 1: Original Sea Level time serie  ===

# === Peaks analysis ===
peak_bt = np.nanmax(bt)
peak_bc = np.nanmax(bc)
peak_bc9 = np.nanmax(bc9)
peak_obs_hourly = np.nanmax(obs_hourly)
peak_obs_hf = np.nanmax(obs_hf)
peak_an9 = np.nanmax(an9)
peak_an9nt = np.nanmax(an9nt)

# === Mean analysis ===
mean_bt         = np.nanmean(bt)
mean_bc         = np.nanmean(bc)
mean_bc9        = np.nanmean(bc9)
mean_obs_1012   = np.nanmean(obs_hourly)
mean_hfobs_1012 = np.nanmean(obs_hf)
mean_an9        = np.nanmean(an9)
mean_an9nt        = np.nanmean(an9nt)

fig, ax = plt.subplots(figsize=(12, 8))
plt.rc('font', size=20)

# Time series
ax.plot(time_bc, bc, color='tab:green', linewidth=1.5, label=f'EAS6 MedFS forecast (12 November peak = {int(peak_bc)} cm, mean ={int(mean_bc)} cm)')
ax.plot(time_bc9, bc9, color='tab:orange', linewidth=1.5, label=f'EAS9 MedFS forecast (12 November peak = {int(peak_bc9)} cm, mean ={int(mean_bc9)} cm)')
ax.plot(time_bt, bt, color='magenta', linewidth=1.5, label=f'EAS9-BT MedFS forecast (12 November peak = {int(peak_bt)} cm, mean ={int(mean_bt)} cm)')
#ax.plot(time_obs_hf, obs_hf, 'b--', linewidth=2, label=f'10 min freq. OBS (12 November peak = {int(peak_obs_hf)} cm, mean ={int(mean_obs_1012)} cm)')
ax.plot(time_obs_hourly, obs_hourly, 'o-', color='navy', linewidth=1.5, label=f'Hourly OBS (12 November peak = {int(peak_obs_hourly)} cm, mean ={int(mean_hfobs_1012)} cm)', markersize=4)
#ax.plot(time_tpxo, tpxo, 'k--', label='Tides TPXO9')

# Extreme threshold
ax.axhline(140, color='red', linewidth=1.5, label='Extreme floods threshold (140 cm)')

# Highlight the 12 novembre 2019 AA event
gray_start = datetime(2019, 11, 12, 18)
gray_end = datetime(2019, 11, 12, 22)
ax.axvspan(gray_start, gray_end, color='grey', alpha=0.3)

# Layout e legenda
ax.set_xlim(start_time, end_time)
ax.set_ylim(-50, 200)
ax.set_ylabel('Sea Level [cm]', fontsize=20)
ax.set_xlabel('Days of November 2019', fontsize=20)
ax.set_title('Original Sea Level observations and MedFS forecast at ISMAR_TG', fontsize=20)

ax.grid(True)
ax.legend(loc='upper left', fontsize=18)

# x-axis
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax.get_xticklabels(which='major'), fontsize=16, weight='bold')
plt.setp(ax.get_xticklabels(which='minor'), fontsize=16, color='gray')
plt.setp(ax.get_yticklabels(), fontsize=16)
ax.grid(which='major', axis='both', linestyle='-', linewidth=0.8, color='black')     # ogni giorno
ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='gray')     # ogni 6 ore
ax.tick_params(axis='x', which='minor', length=4)
ax.grid(True, which='both')

# Save the plot
plt.tight_layout()
plt.savefig('sea_level_ISMAR_TG_originalplot.png', dpi=300)


# === Plot 1b: Original Sea Level time serie  ===

# === Peaks analysis ===
peak_bc9_20191111 = np.nanmax(bc9_20191111)
peak_bc9_20191112 = np.nanmax(bc9_20191112)
peak_bc9_20191113 = np.nanmax(bc9_20191113)
peak_bc9_20191114 = np.nanmax(bc9_20191114)
peak_bc9_20191115 = np.nanmax(bc9_20191115)

# === Mean analysis ===
mean_bc9_20191111        = np.nanmean(bc9_20191111)
mean_bc9_20191112        = np.nanmean(bc9_20191112)
mean_bc9_20191113        = np.nanmean(bc9_20191113)
mean_bc9_20191114        = np.nanmean(bc9_20191114)
mean_bc9_20191115        = np.nanmean(bc9_20191115)

fig, ax = plt.subplots(figsize=(20, 6))
plt.rc('font', size=14)

# Time series
#ax.plot(time_bc, bc, color='tab:green', linewidth=1.5, label=f'EAS6 MedFS forecast (12 November peak = {int(peak_bc)} cm, mean ={int(mean_bc)} cm)')
ax.plot(time_an9, an9, color='black', linewidth=3, label=f'EAS9 MedFS analysis (12 November peak = {int(peak_an9)} cm, mean ={int(mean_an9)} cm)')
ax.plot(time_an9nt, an9nt, color='black', linewidth=3, label=f'EAS9 MedFS analysis (12 November peak = {int(peak_an9nt)} cm, mean ={int(mean_an9nt)} cm)')
#colors = cm.get_cmap('Oranges')(np.linspace(0.4, 0.9, 6))
colors = cm.get_cmap('gist_rainbow_r')(np.linspace(0, 1, 6))
ax.plot(time_bc9, bc9, color=colors[0], linewidth=3, label=f'EAS9 MedFS forecast 20191110 (12 November peak = {int(peak_bc9)} cm, mean ={int(mean_bc9)} cm)')
ax.plot(time_bc9_20191111, bc9_20191111, color=colors[1], linewidth=3, label=f'EAS9 MedFS forecast 20191111 (12 November peak = {int(peak_bc9_20191111)} cm, mean ={int(mean_bc9_20191111)} cm)')
ax.plot(time_bc9_20191112, bc9_20191112, color=colors[2], linewidth=3, label=f'EAS9 MedFS forecast 20191112 (12 November peak = {int(peak_bc9_20191112)} cm, mean ={int(mean_bc9_20191112)} cm)')
ax.plot(time_bc9_20191113, bc9_20191113, color=colors[3], linewidth=3, label=f'EAS9 MedFS forecast 20191113 (mean ={int(mean_bc9_20191113)} cm)')
ax.plot(time_bc9_20191114, bc9_20191114, color=colors[4], linewidth=3, label=f'EAS9 MedFS forecast 20191114 (mean ={int(mean_bc9_20191114)} cm)')
ax.plot(time_bc9_20191115, bc9_20191115, color=colors[5], linewidth=3, label=f'EAS9 MedFS forecast 20191115 (mean ={int(mean_bc9_20191115)} cm)')
#ax.plot(time_bt, bt, color='magenta', linewidth=1.5, label=f'EAS9-BT MedFS forecast (12 November peak = {int(peak_bt)} cm, mean ={int(mean_bt)} cm)')
#ax.plot(time_obs_hf, obs_hf, 'b--', linewidth=2, label=f'10 min freq. OBS (12 November peak = {int(peak_obs_hf)} cm, mean ={int(mean_obs_1012)} cm)')
ax.plot(time_obs_hourly, obs_hourly, 'o-', color='navy', linewidth=2, label=f'Hourly OBS (12 November peak = {int(peak_obs_hourly)} cm, mean ={int(mean_hfobs_1012)} cm)', markersize=4)
ax.plot(time_tpxo, tpxo, 'k--', label='Tides TPXO9')

# Extreme threshold
ax.axhline(140, color='red', linestyle='--',linewidth=1.5, label='Extreme floods threshold (140 cm)')

# Highlight the 12 novembre 2019 AA event
gray_start = datetime(2019, 11, 12, 18)
gray_end = datetime(2019, 11, 12, 22)
ax.axvspan(gray_start, gray_end, color='grey', alpha=0.3)

# Layout e legenda
ax.set_xlim(start_time, end_time)
ax.set_ylim(-80, 200)
ax.set_ylabel('Sea Level [cm]', fontsize=14)
ax.set_xlabel('Days of November 2019', fontsize=14)
ax.set_title('Original Sea Level observations and MedFS forecast at ISMAR_TG', fontsize=14)

ax.grid(True)
ax.legend(loc='upper right', fontsize=14)

# x-axis
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax.get_xticklabels(which='major'), fontsize=14, weight='bold')
plt.setp(ax.get_xticklabels(which='minor'), fontsize=12, color='gray')
plt.setp(ax.get_yticklabels(), fontsize=14)
ax.grid(which='major', axis='both', linestyle='-', linewidth=0.8, color='black')     # ogni giorno
ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='gray')     # ogni 6 ore
ax.tick_params(axis='x', which='minor', length=4)
ax.grid(True, which='both')

# Save the plot
plt.tight_layout()
plt.savefig('sea_level_ISMAR_TG_originalplot_FC.png', dpi=300)

# === Plot 2: Sea Level + offset ===

# Store the original values:
bt_or=bt
bc_or=bc
bc9_or=bc9
an9_or=an9
time_bt_or=time_bt
time_bc_or=time_bc
time_bc9_or=time_bc9
time_an9_or=time_an9

# === Type of offset ===
for offset_type in range (8):
   print (' ')

   # Restore the original values: 
   bt=bt_or
   bc=bc_or
   bc9=bc9_or
   time_bt=time_bt_or
   time_bc=time_bc_or
   time_bc9=time_bc9_or

   # === Mean analysis and offset setting ===
   # Set the model offset
   if offset_type == 0 :
      print ('Offset -> first value')
      mean_4_offset =  obs_hourly[0]
      print ('Obs mean:',mean_4_offset)
   
      bt = bt - bt[0] + mean_4_offset  
      bc = bc - bc[0] + mean_4_offset 
      bc9 = bc9 - bc9[0] + mean_4_offset 
      an9 = an9 - an9[0] + mean_4_offset
      obs_hourly = obs_hourly 
      obs_hf = obs_hf 
      tpxo = tpxo - np.nanmean(tpxo) + mean_4_offset
      print ('MedFS (EAS6,EAS9) 0:', bc[0], bc9[0])
      print ('BT MedFS 0:', bt[0])
      print (' ')
   
   elif offset_type == 1 :
      print ('Offset -> obs mean on ',start_time,'-',end_time,' period')
      mean_4_offset =  np.nanmean(obs_hourly) 
      print ('Obs mean:',mean_4_offset)
      print ('EAS6 MedFS mean:',np.nanmean(bc))
      print ('EAS9 MedFS mean:',np.nanmean(bc9))
      print ('BT MedFS mean:',np.nanmean(bt))
      print (' ')
   
      bt = bt - np.nanmean(bt) + mean_4_offset  
      bc = bc - np.nanmean(bc) + mean_4_offset  
      bc9 = bc9 - np.nanmean(bc9) + mean_4_offset
      an9 = an9 - np.nanmean(an9) + mean_4_offset
      obs_hourly = obs_hourly 
      obs_hf = obs_hf 
      tpxo = tpxo - np.nanmean(tpxo) + mean_4_offset
   
   elif offset_type == 2 :
      print ('Offset ->  mod - nov mod mean + 10-13 obs mean (paper)')
      obs_paper_mean   =  0.76     # OBS Mean at ISMAR_TG = 0.76 (10-15 Nov) or  = 0.71 (10-12 Nov)
      mod_mean6        = -0.0995   # Nov 2019 Mean at ISMAR TG
      mod_meanbt       = 0.2318    # Nov 2019 Mean at ISMAR_TG
      mod_mean9        = -0.12536  # Nov 2019 Mean at ISMAR_TG
      tpxo_mean        = -0.0004   # Nov 2019 Mean at ISMAR_TG
   
      # m -> cm
      mean_4_offset =  obs_paper_mean*100
      print ('Obs mean:',mean_4_offset)
      mod_mean6  = mod_mean6*100
      mod_meanbt = mod_meanbt*100
      mod_mean9  = mod_mean9*100  
      tpxo_mean  = tpxo_mean*100
      print ('EAS6 MedFS mean:',mod_mean6)
      print ('EAS9 MedFS mean:',mod_mean9)
      print ('BT MedFS mean:',mod_meanbt)   
      print (' ')

      bt = bt - mod_meanbt + mean_4_offset
      bc = bc - mod_mean6 + mean_4_offset
      bc9 = bc9 - mod_mean9 + mean_4_offset
      an9 = an9 - np.nanmean(an9) + mean_4_offset
      obs_hourly = obs_hourly
      obs_hf = obs_hf
      tpxo = tpxo - np.nanmean(tpxo) + mean_4_offset
   
   elif offset_type == 3 :
      print ('Offset -> 1 week before FC init')
      mean_4_offset =  obs_Nov_4_10
      print ('Obs mean:',mean_4_offset)
      print ('EAS6 MedFS mean:',np.nanmean(bc))
      print ('EAS9 MedFS mean:',np.nanmean(bc9))
      print ('BT MedFS mean:',np.nanmean(bt))
      print (' ')
   
      bt = bt - np.nanmean(bt) + mean_4_offset
      bc = bc - np.nanmean(bc) + mean_4_offset
      bc9 = bc9 - np.nanmean(bc9) + mean_4_offset
      an9 = an9 - np.nanmean(an9) + mean_4_offset
      obs_hourly = obs_hourly
      obs_hf = obs_hf
      tpxo = tpxo - np.nanmean(tpxo) + mean_4_offset
   
   elif offset_type == 4 :
      print ('Offset -> 1 month before FC init')
      mean_4_offset =  obs_10Oct_10Nov_2019
      print ('Obs mean:',mean_4_offset)
      print ('EAS6 MedFS mean:',np.nanmean(bc))
      print ('EAS9 MedFS mean:',np.nanmean(bc9))
      print ('BT MedFS mean:',np.nanmean(bt))
      print (' ')
   
      bt = bt - np.nanmean(bt) + mean_4_offset
      bc = bc - np.nanmean(bc) + mean_4_offset
      bc9 = bc9 - np.nanmean(bc9) + mean_4_offset
      an9 = an9 - np.nanmean(an9) + mean_4_offset
      obs_hourly = obs_hourly
      obs_hf = obs_hf
      tpxo = tpxo - np.nanmean(tpxo) + mean_4_offset
   
   elif offset_type == 5 :
      print ('Offset -> mod - 10-13 mod mean + 11-17 obs mean')
      mean_4_offset =  obs_Nov_11_17
      print ('Obs mean:',mean_4_offset)
      print ('EAS6 MedFS mean:',np.nanmean(bc))
      print ('EAS9 MedFS mean:',np.nanmean(bc9))
      print ('BT MedFS mean:',np.nanmean(bt))
      print (' ')

      bt = bt - np.nanmean(bt) + mean_4_offset
      bc = bc - np.nanmean(bc) + mean_4_offset
      bc9 = bc9 - np.nanmean(bc9) + mean_4_offset
      an9 = an9 - np.nanmean(an9) + mean_4_offset
      obs_hourly = obs_hourly
      obs_hf = obs_hf
      tpxo = tpxo - np.nanmean(tpxo) + mean_4_offset
   
   elif offset_type == 6 :
      print ('Offset ->slides bajo and emanuela: mod - 10-13 mod mean + 10 12:00 - 13 12:00 obs mean')
      mean_4_offset = obs_Nov_10_125
      print ('Obs mean:',mean_4_offset)
      print ('EAS6 MedFS mean:',np.nanmean(bc))
      print ('EAS9 MedFS mean:',np.nanmean(bc9))
      print ('BT MedFS mean:',np.nanmean(bt))
      print (' ')
   
      bt = bt - np.nanmean(bt) + mean_4_offset
      bc = bc - np.nanmean(bc) + mean_4_offset
      bc9 = bc9 - np.nanmean(bc9) + mean_4_offset
      an9 = an9 - np.nanmean(an9) + mean_4_offset
      obs_hourly = obs_hourly
      obs_hf = obs_hf
      tpxo = tpxo - np.nanmean(tpxo) + mean_4_offset
   
   elif offset_type == 7 :
      print ('Offset -> mod - Nov mod mean + Nov obs mean ')
      mean_4_offset = obs_Nov_2019
      print ('Obs mean:',mean_4_offset)
      mod_mean6        = -0.0995   # Nov 2019 Mean at ISMAR TG
      mod_meanbt       = 0.2318    # Nov 2019 Mean at ISMAR_TG
      mod_mean9        = -0.12536  # Nov 2019 Mean at ISMAR_TG
      print ('EAS6 MedFS mean:',np.nanmean(bc))
      print ('EAS9 MedFS mean:',np.nanmean(bc9))
      print ('BT MedFS mean:',np.nanmean(bt))
      print (' ')

      bt = bt - np.nanmean(mod_meanbt) + mean_4_offset
      bc = bc - np.nanmean(mod_mean6) + mean_4_offset
      bc9 = bc9 - np.nanmean(mod_mean9) + mean_4_offset
      an9 = an9 - np.nanmean(mod_mean9) + mean_4_offset
      obs_hourly = obs_hourly
      obs_hf = obs_hf
      tpxo = tpxo - np.nanmean(tpxo) + mean_4_offset

   # === Peaks analysis ===
   peak_bt = np.nanmax(bt)
   peak_bc = np.nanmax(bc)
   peak_bc9 = np.nanmax(bc9)
   peak_an9 = np.nanmax(an9)
   peak_obs_hourly = np.nanmax(obs_hourly)
   peak_obs_hf = np.nanmax(obs_hf)
   
   fig, ax = plt.subplots(figsize=(12, 8))
   plt.rc('font', size=20)
   
   # Time series
   #ax.plot(time_an9, an9, color='cyan', linewidth=1.5, label=f'EAS9 MedFS analysis (12 November peak = {int(peak_an9)} cm)')
   #
   ax.plot(time_obs_hf, obs_hf, 'b--', linewidth=3, label=f'10 min freq. OBS (12 November peak = {int(peak_obs_hf)} cm)')
   ax.plot(time_obs_hourly, obs_hourly, 'o-', color='navy', linewidth=3, label=f'Hourly OBS (12 November peak = {int(peak_obs_hourly)} cm)', markersize=4)
   #
   #ax.plot(time_bc, bc, linestyle='-', color='magenta', linewidth=3, label=f'EAS6 MedFS forecast (12 November peak = {int(peak_bc)} cm)')
   ax.plot(time_bc9, bc9, color='tab:green', linewidth=3, label=f'EAS9 MedFS forecast (12 November peak = {int(peak_bc9)} cm)')
   ax.plot(time_bt, bt, color='tab:orange',linewidth=3, label=f'MedFS BT forecast (12 November peak = {int(peak_bt)} cm)')
   #
   ax.plot(time_tpxo, tpxo, 'k--',linewidth=2,label='Tides TPXO9')
   
   # Extreme threshold 
   ax.axhline(140, color='red', linewidth=2, label='Extreme floods threshold (140 cm)')
   # Add mean obs offset
   plt.axhline(mean_4_offset,color='black',linewidth=2,label='Mean OBS ('+str(round(mean_4_offset))+' cm)',zorder=0)

   # Highlight the 12 novembre 2019 AA event
   gray_start = datetime(2019, 11, 12, 18)
   gray_end = datetime(2019, 11, 12, 22)
   ax.axvspan(gray_start, gray_end, color='grey', alpha=0.3)
   
   # Layout e legenda
   ax.set_xlim(start_time, end_time)
   ax.set_ylim(0, 200)
   ax.set_ylabel('Sea Level [cm]', fontsize=20)
   ax.set_xlabel('Days of November 2019', fontsize=20)
   ax.set_title('Sea Level observations and MedFS forecast at ISMAR_TG', fontsize=20)
   
   ax.grid(True)
   ax.legend(loc='upper left', fontsize=16, framealpha=0.9)
   
   # x-axis
   ax.xaxis.set_major_locator(mdates.DayLocator())
   ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
   ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
   ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
   plt.setp(ax.get_xticklabels(which='major'), fontsize=16, weight='bold')
   plt.setp(ax.get_xticklabels(which='minor'), fontsize=16, color='gray')
   plt.setp(ax.get_yticklabels(), fontsize=16)
   ax.grid(which='major', axis='both', linestyle='-', linewidth=0.8, color='black')     # ogni giorno
   ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='gray')     # ogni 6 ore
   ax.tick_params(axis='x', which='minor', length=4)
   ax.grid(True, which='both')
   
   # Save the plot
   plt.tight_layout()
   plt.savefig('sea_level_ISMAR_TG_newplot_'+str(offset_type)+'_EAS9_EAS9BT.png', dpi=300)

# === Plot 3: Sea Level Anomaly ===

# Anomaly time-series
bt = bt-np.nanmean(bt)
bc = bc-np.nanmean(bc)
bc9 = bc9-np.nanmean(bc9)
an9 = an9-np.nanmean(an9)
obs_hourly = obs_hourly-np.nanmean(obs_hourly)
obs_hf = obs_hf-np.nanmean(obs_hf)

# Anomaly peaks
peak_bt = np.nanmax(bt)
peak_bc = np.nanmax(bc)
peak_bc9 = np.nanmax(bc9)
peak_an9 = np.nanmax(an9)
peak_obs_hourly = np.nanmax(obs_hourly)
peak_obs_hf = np.nanmax(obs_hf)

fig, ax = plt.subplots(figsize=(12, 8))
plt.rc('font', size=20)

# Time series
ax.plot(time_bc, bc, color='tab:green', linewidth=1.5, label=f'MedFS forecast (12 November peak = {int(peak_bc)} cm)')
ax.plot(time_bc9, bc9, color='tab:orange', linewidth=1.5, label=f'MedFS forecast (12 November peak = {int(peak_bc9)} cm)')
ax.plot(time_an9, an9, color='tab:orange', linewidth=1.5, label=f'MedFS forecast (12 November peak = {int(peak_an9)} cm)')
ax.plot(time_bt, bt, color='magenta', linewidth=1.5, label=f'EAS9-BT MedFS forecast (12 November peak = {int(peak_bt)} cm)')
#ax.plot(time_obs_hf, obs_hf, 'b--', linewidth=2, label=f'10 min freq. OBS (12 November peak = {int(peak_obs_hf)} cm)')
ax.plot(time_obs_hourly, obs_hourly, 'o-', color='navy', linewidth=1.5, label=f'Hourly OBS (12 November peak = {int(peak_obs_hourly)} cm)', markersize=4)
#ax.plot(time_tpxo, tpxo, 'k--', label='Tides TPXO9')

# Higlight the 0 val
ax.axhline(0, color='black', linewidth=2.5, label='')

# Highlight the 12 novembre 2019 AA event
gray_start = datetime(2019, 11, 12, 18)
gray_end = datetime(2019, 11, 12, 22)
ax.axvspan(gray_start, gray_end, color='grey', alpha=0.3)

# Layout e legenda
ax.set_xlim(start_time, end_time)
ax.set_ylim(-125, 125)
ax.set_ylabel('Sea Level Anomaly [cm]', fontsize=20)
ax.set_xlabel('Days of November 2019', fontsize=20)
ax.set_title('Sea Level anomaly (observations and MedFS forecast) at ISMAR_TG', fontsize=20)

ax.grid(True)
ax.legend(loc='upper left', fontsize=18)

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax.get_xticklabels(which='major'), fontsize=16, weight='bold')
plt.setp(ax.get_xticklabels(which='minor'), fontsize=16, color='gray')
plt.setp(ax.get_yticklabels(), fontsize=16)
ax.grid(which='major', axis='both', linestyle='-', linewidth=0.8, color='black')     # ogni giorno
ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='gray')     # ogni 6 ore
ax.tick_params(axis='x', which='minor', length=4)
ax.grid(True, which='both')

# Save the plot
plt.tight_layout()
plt.savefig('sea_level_ISMAR_TG_anomalyplot.png', dpi=300)

