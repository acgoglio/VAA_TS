#
# Script for HARMONIC ANALYSIS AND POST_PROC
#
# imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl # Palettes
import numpy as np
import netCDF4 as NC
import os
import sys
import warnings
#
warnings.filterwarnings("ignore") # Avoid warnings
#
import scipy as sp
from scipy import signal
from scipy.optimize import curve_fit
from scipy import stats
import collections
import pandas as pd
import csv
import math
import datetime
from datetime import datetime, timedelta
from operator import itemgetter
import plotly
from plotly import graph_objects as go # for bar plot
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from operator import itemgetter # to order lists
from statsmodels.distributions.empirical_distribution import ECDF # empirical distribution functions
#
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pylab as pl
#
import tidal_filter as mod
mpl.use('Agg')
#
workdir='/work/cmcc/ag15419/VAA_EAS9/' 
outfile=open(workdir+'bandwidths.txt',"w")
coo_file='AAV_adr_2022.coo'
model_bathy='/work/cmcc/ag15419/VAA_paper/DATA0/bathy_meter.nc'
colors=['red','slategrey','darkgreen','tab:green','lime','black']
############################
# Set the bandwidth
# SEICHES: LOW: 11.1h / HIGH: 21.1h
lowf_seiches_band=[9.9,11.5] # Anna Chiara's values: [10,12.2] # Jacopo's values: [9.5,12.2] # choosen values [9.9,11.5] # choosen by AC values [10.3,11.3]
highf_seiches_band=[17.5,22.3] #Anna Chiara's values: [17.2,23.2] # Jacopo's values: [16.7,24.2] # choosen values [17.5,22.3] # choosen by AC values [19,21.7]
#
# TIDES: SEMID: 12.4h,12.0h,12.7h12.0h / DIURNAL:23.9h,25.8h,24.1h,26.9h
semid_tides_band=[11.2,14.2] #Anna Chiara's values: [11.2,16] # Jacopo's values: [11.2,14.2] # choosen values [11.2,16] # choosen by AC values [11.2,16]
diurnal_tides_band=[21.7,30] #Anna Chiara's values: [21.7,32] # Jacopo's values: [21.7,30.0] # choosen values [21.7,32] # choosen by AC values [21.7,32]


print ('LOW seiches bandwidth [h]',lowf_seiches_band[0],lowf_seiches_band[1],file=outfile)
print ('HIGH seiches bandwidth [h]',highf_seiches_band[0],highf_seiches_band[1],file=outfile)
print ('LOW tides bandwidt [h]',diurnal_tides_band[0],diurnal_tides_band[1],file=outfile)
print ('HIGH tides bandwidth [h]',semid_tides_band[0],semid_tides_band[1],file=outfile)

# Compute total bandwidth
lowf_all_band=[min(lowf_seiches_band[0],semid_tides_band[0]),max(lowf_seiches_band[1],semid_tides_band[1])]
highf_all_band=[min(highf_seiches_band[0],diurnal_tides_band[0]),max(highf_seiches_band[1],diurnal_tides_band[1])]

print ('LOW total bandwidth',lowf_all_band[0],lowf_all_band[1],file=outfile)
print ('HIGH total bandwidth',highf_all_band[0],highf_all_band[1],file=outfile)

post_name='_all'

#ALL_STGS=['ISMAR_TG','RMN-Venice','RMN-Ravenna','RMN-Ancona','RMN-Trieste','RMN-SBenedettoDelTronto','RMN-Ortona','RMN-Vieste','RMN-Otranto']
#['ISMAR_TG','RMN-Venice','RMN-Ravenna','RMN-Ancona','RMN-Trieste','RMN-SBenedettoDelTronto','RMN-Ortona','RMN-Vieste','RMN-Otranto']:

# Open  coo file and get values
fT1_coo = pd.read_csv(coo_file,sep=';',comment='#',header=None)
# lat ; lon ; name ;
tg_lat = fT1_coo[0][:]
tg_lon = fT1_coo[1][:]
ALL_STGS = fT1_coo[2][:]


for STG in ALL_STGS:
   print ('TG: ',STG,file=outfile)
   print ('Working on TG: ',STG)

   # save diagnostic vars to be plotted
   globals()['diag_totmax'+str(STG)]=[]
   globals()['diag_tidesmax'+str(STG)]=[]
   globals()['diag_seichesmax'+str(STG)]=[]
   globals()['diag_totmean'+str(STG)]=[]
   globals()['diag_tidesmean'+str(STG)]=[]
   globals()['diag_seichesmean'+str(STG)]=[]
   globals()['diag_totstd'+str(STG)]=[]
   globals()['diag_tidesstd'+str(STG)]=[]
   globals()['diag_seichesstd'+str(STG)]=[]
   # compute statistic vars to be plotted
   globals()['diag_summax'+str(STG)]=[]
   globals()['diag_summean'+str(STG)]=[]
   globals()['diag_sumstd'+str(STG)]=[]
   
   if STG == 'ISMAR_TG':
      # For this case the obs are in csv format and at 00
      # thus also the time-series from the models have been interpolated at 00
      non_tidal=workdir+'ISMAR_TG_mod_EAS9-NT.nc'
      tidal1=workdir+STG+'_mod_eas6_all1_long.nc' 
      tidal2=workdir+'ISMAR_TG_mod_EAS9.nc'
      tidal3=workdir+'ISMAR_TG_mod_EAS9-simu.nc'
      obs_emodnet=tidal1 #workdir+'RMN-Venice_25h_obs_nondet_00.nc'
      obs_ismar=workdir+'piattaforma2022_long_20Oct_22Nov.csv'
      obs_ismar_ref=workdir+'piattaforma2022_long_6Nov_17Nov.csv'
      time_csv=workdir+'piattaforma2019_false_long.csv'
   else:
      non_tidal=workdir+'RMN-Venice_mod_EAS9-NT.nc'
      tidal1=workdir+'RMN-Venice_mod_eas6_fc.nc'
      tidal2=workdir+'RMN-Venice_mod_EAS9.nc'
      tidal3=workdir+'RMN-Venice_mod_EAS9-simu.nc'
      obs_emodnet=workdir+'MO_TS_TG_RMN-Venice_20Oct-22Nov.nc'
      obs_emodnet_ref=obs_emodnet
      obs_ismar=workdir+'piattaforma2022_long_20Oct_22Nov.csv'
      obs_ismar_ref=workdir+'piattaforma2022_long_20Oct_17Nov.csv'
      time_csv=workdir+'piattaforma2019_false_long.csv'
   
   ############################
   # Read mod 
   fT1_mod_nontidal = NC.Dataset(non_tidal,'r')
   #fT1_mod_nontidal2 = NC.Dataset(non_tidal2,'r')
   fT1_mod_detided  = NC.Dataset(tidal1,'r')
   fT1_mod_tidal    = fT1_mod_detided
   fT2_mod_detided  = NC.Dataset(tidal2,'r')
   fT2_mod_tidal    = fT2_mod_detided
   fT3_mod_detided  = NC.Dataset(tidal3,'r')
   fT3_mod_tidal    = fT3_mod_detided
   
   mod_nontidal = fT1_mod_nontidal.variables['sossheig'][:,0,0] *100.0 # want cm not meters
   mod_nontidal = np.array(mod_nontidal)
   #mod_nontidal2 = fT1_mod_nontidal2.variables['zos'][:,0,0] *100.0 # want cm not meters
   #mod_nontidal2 = np.array(mod_nontidal2)
   
   mod_tidal1 = fT1_mod_tidal.variables['zos'][:,0,0] *100.0 # want cm not meters
   mod_tidal1 = np.array(mod_tidal1)

   mod_tidal2 = fT2_mod_tidal.variables['sossheig'][:,0,0] *100.0 # want cm not meters
   mod_tidal2 = np.array(mod_tidal2)

   mod_tidal3 = fT3_mod_tidal.variables['sossheig'][:,0,0] *100.0 # want cm not meters
   mod_tidal3 = np.array(mod_tidal3)   

   mod_detided1 = fT1_mod_detided.variables['zos_detided'][:,0,0] *100.0 # want cm not meters
   mod_detided1 = np.array(mod_detided1)

   mod_detided2 = fT1_mod_detided.variables['zos_detided'][:,0,0] *100.0 # want cm not meters
   mod_detided2 = np.array(mod_detided2)

   mod_detided3 = fT1_mod_detided.variables['zos_detided'][:,0,0] *100.0 # want cm not meters
   mod_detided3 = np.array(mod_detided3)

   mod_nontidal1 = mod_detided1 # TMP
   mod_nontidal2 = mod_detided1 # TMP
   mod_nontidal3 = mod_detided1 # TMP
   
   ## Read EMODnet OBS
   if STG == 'ISMAR_TG':
      fT1_obs_emodnet = NC.Dataset(obs_emodnet,'r')
   if STG != 'ISMAR_TG':
      fT1_obs_emodnet = NC.Dataset(obs_emodnet,'r')
      fT1_obs_emodnet_ref = NC.Dataset(obs_emodnet_ref,'r')
      obs_emodnet = fT1_obs_emodnet.variables['SLEV'][:]*100.0 # want cm not meters
      ##
      obs_emodnet = np.array(obs_emodnet)
      #obs_emodnet=obs_emodnet[:,0]
      obs_emodnet=np.squeeze(obs_emodnet)
      obs_emodnet_ref=np.squeeze(obs_emodnet)
   #
   # Read ISMAR obs
   if STG == 'ISMAR_TG':
      fTt_time = pd.read_csv(time_csv,sep=';',comment='#',header=None)
      fT2_coo = pd.read_csv(obs_ismar,sep=';',comment='#',header=None)
      fT3_coo = pd.read_csv(obs_ismar_ref,sep=';',comment='#',header=None)
      #
      #ismar_data_time = fTt_time[0][:]
      min_obs_ismar = fT2_coo[1][:]*100.0
      min_obs_ismar = np.array(min_obs_ismar)
      min_obs_ismar_ref = fT3_coo[1][:]*100.0
      min_obs_ismar_ref = np.array(min_obs_ismar_ref)
      steps = 12
      obs_ismar=[]
      obs_ismar_ref=[]
      for i in range(0, len(min_obs_ismar), steps):
          #obs_ismar.append(np.average(min_obs_ismar[i:i+steps]))
          obs_ismar.append(np.nanmean(min_obs_ismar[i:i+steps]))
      obs_ismar = np.array(obs_ismar)
      for i in range(0, len(min_obs_ismar_ref), steps):
          obs_ismar_ref.append(np.nanmean(min_obs_ismar_ref[i:i+steps]))
      obs_ismar_ref = np.array(obs_ismar_ref)

   else:
      obs_ismar = obs_emodnet
      obs_ismar_ref = obs_emodnet_ref 
      print ('Prova obs_emodnet',obs_ismar)

      ## Read also the wind
      #ismar_wind_dir =  np.array(fT2_coo[2][:])
      #ismar_wind_spe =  np.array(fT2_coo[3][:])
      #ismar_wind_max =  np.array(fT2_coo[4][:])
#   else:
#      obs_ismar    = obs_emodnet[0:-1]
#      obs_emodnet  = obs_emodnet[0:-1]
#      mod_nontidal = mod_nontidal[0:-1]
#     #mod_nontidal2 = mod_nontidal2[0:-1]
#     mod_tidal    = mod_tidal[0:-1]
#      mod_detided  = mod_detided[0:-1]
#      # Mask the nans values in all the datasets (to be compared)
#      mask_obs = ~np.isnan(obs_ismar) | ~np.isnan(obs_emodnet)
#      obs_ismar    = obs_ismar[mask_obs]
#      obs_emodnet  = obs_emodnet[mask_obs]
#      mod_nontidal = mod_nontidal[mask_obs]
#      #mod_nontidal2 = mod_nontidal2[mask_obs]
#      mod_tidal    = mod_tidal[mask_obs]
#      mod_detided  = mod_detided[mask_obs]

   # COMPUTE THE TPXO TIDAL SIGNAL
   tpxo=mod_tidal1-mod_detided1
   #
   # Detide OBS by means of TPXO
   ##obs_emodnet_det = obs_emodnet - tpxo
   ##obs_ismar_det = obs_ismar - tpxo
   #
   # Detide OBS by means of TPXO
   #obs_emodnet_det = obs_emodnet - tpxo
   if len(tpxo) > len(obs_ismar):
      obs_ismar_det = obs_ismar - tpxo[-len(obs_ismar):]
   else:
      obs_ismar_det = obs_ismar[-len(tpxo):] - tpxo
   #print ('Prova1 obs ',np.max(obs_ismar),np.min(obs_ismar))
   #print ('Prova2 mod ',np.max(mod_tidal),np.min(mod_tidal))
   #print ('Prova len obs mod ',len(obs_ismar),len(mod_tidal))


   # Apply the FFT filter to the mod and obs
   
   # Extract tides
   tides_mod_tidal1_fft,resp_tides1 = mod.fft_2bands(mod_tidal1,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='True')
   tides_mod_tidal2_fft,resp_tides2 = mod.fft_2bands(mod_tidal2,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='True')
   tides_mod_tidal3_fft,resp_tides3 = mod.fft_2bands(mod_tidal3,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='True')
   tides_obs_ismar_fft,resp_tides = mod.fft_2bands(obs_ismar,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='True')
   # Remove tides
   residual_mod_tidal1_fft,resp_res1 = mod.fft_2bands(mod_tidal1,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='False')
   residual_mod_tidal2_fft,resp_res2 = mod.fft_2bands(mod_tidal2,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='False')
   residual_mod_tidal3_fft,resp_res3 = mod.fft_2bands(mod_tidal3,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='False')
   residual_obs_ismar_fft,resp_res = mod.fft_2bands(obs_ismar,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='False')
   residual_mod_nontidal_fft,resp_resn = mod.fft_2bands(mod_nontidal,low_bound=1./(diurnal_tides_band[0]), high_bound=1./(diurnal_tides_band[1]),low_bound_1=1/(semid_tides_band[0]), high_bound_1=1/(semid_tides_band[1]),alpha=0.4,invert='False')
   # Extract seiches
   seiches_mod_tidal1_fft,resp_seiches1 = mod.fft_2bands(mod_tidal1,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')
   seiches_mod_tidal2_fft,resp_seiches2 = mod.fft_2bands(mod_tidal2,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')
   seiches_mod_tidal3_fft,resp_seiches3 = mod.fft_2bands(mod_tidal3,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')
   seiches_mod_nontidal_fft,resp_seichesn = mod.fft_2bands(mod_nontidal,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')
   seiches_mod_nontidal1_fft,resp_seiches1 = mod.fft_2bands(mod_nontidal1,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')
   seiches_mod_nontidal2_fft,resp_seiches2 = mod.fft_2bands(mod_nontidal2,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')
   seiches_mod_nontidal3_fft,resp_seiches3 = mod.fft_2bands(mod_nontidal3,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')
   #seiches_mod_nontidal2_fft,resp_seiches = mod.fft_2bands(mod_nontidal2,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')
   seiches_obs_ismar_fft,resp_seiches = mod.fft_2bands(obs_ismar_det,low_bound=1./(highf_seiches_band[0]), high_bound=1./(highf_seiches_band[1]),low_bound_1=1/(lowf_seiches_band[0]), high_bound_1=1/(lowf_seiches_band[1]),alpha=0.4,invert='True')

   # Remove seiches and/or tides
   residual_all_mod_tidal1_fft,resp_all_res1 = mod.fft_2bands(mod_tidal1,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   residual_all_mod_tidal2_fft,resp_all_res2 = mod.fft_2bands(mod_tidal2,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   residual_all_mod_tidal3_fft,resp_all_res3 = mod.fft_2bands(mod_tidal3,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   residual_all_mod_nontidal_fft,resp_all_resn = mod.fft_2bands(mod_nontidal,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   residual_all_mod_nontidal1_fft,resp_all_res1 = mod.fft_2bands(mod_nontidal1,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   residual_all_mod_nontidal2_fft,resp_all_res2 = mod.fft_2bands(mod_nontidal2,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   residual_all_mod_nontidal3_fft,resp_all_res3 = mod.fft_2bands(mod_nontidal3,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   residual_all_mod_nontidal2_fft,resp_all_res = mod.fft_2bands(mod_nontidal2,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   residual_all_obs_ismar_fft,resp_all_res = mod.fft_2bands(obs_ismar,low_bound=1./(highf_all_band[0]), high_bound=1./(highf_all_band[1]),low_bound_1=1/(lowf_all_band[0]), high_bound_1=1/(lowf_all_band[1]),alpha=0.4,invert='False')
   
   # Compute the offset value of ISMAR obs before detrending
   obs_ref = np.nanmean(obs_ismar_ref)
   obs_ref_res = np.nanmean(residual_all_obs_ismar_fft)
   obs_ref_tides = np.nanmean(tides_obs_ismar_fft)
   obs_ref_seiches = np.nanmean(seiches_obs_ismar_fft)
   print ('obs_ref/obs_ref_res/obs_ref_tides/obs_ref_seiches',obs_ref,obs_ref_res,obs_ref_tides,obs_ref_seiches)
   
   # Move all the curves on the obs mean value
   
   # Sea Level
#   #obs_emodnet_sealevel = obs_emodnet - np.nanmean(obs_emodnet) + obs_ref
   obs_ismar_sealevel = obs_ismar - np.nanmean(obs_ismar) + obs_ref
   mod_tidal1_sealevel = mod_tidal1 - np.nanmean(mod_tidal1) + obs_ref
   mod_tidal2_sealevel = mod_tidal2 - np.nanmean(mod_tidal2) + obs_ref
   mod_tidal3_sealevel = mod_tidal3 - np.nanmean(mod_tidal3) + obs_ref
#   mod_nontidal1_sealevel = mod_nontidal1 - np.nanmean(mod_nontidal1) + obs_ref
#   mod_nontidal2_sealevel = mod_nontidal2 - np.nanmean(mod_nontidal2) + obs_ref
#   mod_nontidal3_sealevel = mod_nontidal3 - np.nanmean(mod_nontidal3) + obs_ref
   mod_nontidal_sealevel = mod_nontidal - np.nanmean(mod_nontidal) + obs_ref

   # Tides
   tpxo_tid=tpxo-np.nanmean(tpxo) #+obs_ref_tides

   #mod_tidal2_sealevel=mod_tidal2_sealevel[-72:]-np.nanmean(mod_tidal2_sealevel[-72:])#+np.nanmean(obs_ismar_sealevel[-72:])
   #mod_tidal3_sealevel=mod_tidal3_sealevel[-72:]-np.nanmean(mod_tidal3_sealevel[-72:])#+np.nanmean(obs_ismar_sealevel[-72:])
   #mod_nontidal_sealevel=mod_nontidal_sealevel[-72:]+tpxo_tid[-72:]-np.nanmean(mod_nontidal_sealevel[-72:]+tpxo_tid[-72:])#+np.nanmean(obs_ismar_sealevel[-72:])


   # Residual
   obs_ismar_res = obs_ismar_det - np.nanmean(obs_ismar_det) + obs_ref_res
   #obs_emodnet_res = obs_emodnet_det - np.nanmean(obs_emodnet_det) + obs_ref_res
   mod_tidal1_res = mod_detided1 - np.nanmean(mod_detided1) + obs_ref_res
   mod_tidal2_res = mod_detided2 - np.nanmean(mod_detided2) + obs_ref_res
   mod_tidal3_res = mod_detided3 - np.nanmean(mod_detided3) + obs_ref_res
   mod_nontidal1_res = mod_nontidal1 - np.nanmean(mod_nontidal1) + obs_ref_res
   mod_nontidal2_res = mod_nontidal2 - np.nanmean(mod_nontidal2) + obs_ref_res
   mod_nontidal4_res = mod_nontidal3 - np.nanmean(mod_nontidal3) + obs_ref_res
   mod_nontidal_res = mod_nontidal - np.nanmean(mod_nontidal) + obs_ref_res
   #
   mod_tidal1_fft_res=residual_mod_tidal1_fft-np.nanmean(residual_mod_tidal1_fft)+obs_ref_res
   mod_tidal2_fft_res=residual_mod_tidal2_fft-np.nanmean(residual_mod_tidal2_fft)+obs_ref_res
   mod_tidal3_fft_res=residual_mod_tidal3_fft-np.nanmean(residual_mod_tidal3_fft)+obs_ref_res
   obs_ismar_fft_res=residual_obs_ismar_fft-np.nanmean(residual_obs_ismar_fft)+obs_ref_res
   #
   mod_tidal1_fft_res_all=residual_all_mod_tidal1_fft-np.nanmean(residual_all_mod_tidal1_fft) +obs_ref_res
   mod_tidal2_fft_res_all=residual_all_mod_tidal2_fft-np.nanmean(residual_all_mod_tidal2_fft) +obs_ref_res
   mod_tidal3_fft_res_all=residual_all_mod_tidal3_fft-np.nanmean(residual_all_mod_tidal3_fft) +obs_ref_res
   mod_nontidal1_fft_res_all=residual_all_mod_nontidal1_fft-np.nanmean(residual_all_mod_nontidal1_fft)+obs_ref_res
   mod_nontidal2_fft_res_all=residual_all_mod_nontidal2_fft-np.nanmean(residual_all_mod_nontidal2_fft)+obs_ref_res
   mod_nontidal3_fft_res_all=residual_all_mod_nontidal3_fft-np.nanmean(residual_all_mod_nontidal3_fft)+obs_ref_res
   mod_nontidal_fft_res_all=residual_all_mod_nontidal_fft-np.nanmean(residual_all_mod_nontidal_fft)+obs_ref_res
   obs_ismar_fft_res_all=residual_all_obs_ismar_fft-np.nanmean(residual_all_obs_ismar_fft)+obs_ref_res

   # Tides
   tpxo_tid=tpxo-np.nanmean(tpxo) #+obs_ref_tides
   mod_tidal1_fft_tid=tides_mod_tidal1_fft-np.nanmean(tides_mod_tidal1_fft) +obs_ref_tides
   mod_tidal2_fft_tid=tides_mod_tidal2_fft-np.nanmean(tides_mod_tidal2_fft) +obs_ref_tides
   mod_tidal3_fft_tid=tides_mod_tidal3_fft-np.nanmean(tides_mod_tidal3_fft) +obs_ref_tides
   #mod_nontidal_fft_tid=tides_mod_nontidal_fft-np.nanmean(tides_mod_nontidal_fft) +obs_ref_tides
   obs_ismar_fft_tid=tides_obs_ismar_fft-np.nanmean(tides_obs_ismar_fft)+obs_ref_tides

   # Seiches
   mod_tidal1_fft_sei=seiches_mod_tidal1_fft-np.nanmean(seiches_mod_tidal1_fft) +obs_ref_seiches
   mod_tidal2_fft_sei=seiches_mod_tidal2_fft-np.nanmean(seiches_mod_tidal2_fft) +obs_ref_seiches
   mod_tidal3_fft_sei=seiches_mod_tidal3_fft-np.nanmean(seiches_mod_tidal3_fft) +obs_ref_seiches
   mod_nontidal1_fft_sei=seiches_mod_nontidal1_fft-np.nanmean(seiches_mod_nontidal1_fft)+obs_ref_seiches
   mod_nontidal2_fft_sei=seiches_mod_nontidal2_fft-np.nanmean(seiches_mod_nontidal2_fft)+obs_ref_seiches
   mod_nontidal3_fft_sei=seiches_mod_nontidal3_fft-np.nanmean(seiches_mod_nontidal3_fft)+obs_ref_seiches
   mod_nontidal_fft_sei=seiches_mod_nontidal_fft-np.nanmean(seiches_mod_nontidal_fft)+obs_ref_seiches
   obs_ismar_fft_sei=seiches_obs_ismar_fft-np.nanmean(seiches_obs_ismar_fft)+obs_ref_seiches
 
   # Read date/time from emodenet obs
   if STG == 'ISMAR_TG':
      time_obs=fT1_obs_emodnet.variables['time'][:]
      time_obs_units = fT1_obs_emodnet.variables['time'].getncattr('units')
   if STG != 'ISMAR_TG':
      time_obs=fT1_obs_emodnet.variables['TIME'][:]
      time_obs_units = fT1_obs_emodnet.variables['TIME'].getncattr('units')
   alltimes_obs=[]
   for alltime_idx in range (0,len(time_obs)):
              alltimes_obs.append(datetime(NC.num2date(time_obs[alltime_idx],time_obs_units).year,NC.num2date(time_obs[alltime_idx],time_obs_units).month,NC.num2date(time_obs[alltime_idx],time_obs_units).day,NC.num2date(time_obs[alltime_idx],time_obs_units).hour,NC.num2date(time_obs[alltime_idx],time_obs_units).minute,NC.num2date(time_obs[alltime_idx],time_obs_units).second))
   
   if STG == 'ISMAR_TG':
      alltimes=alltimes_obs
      alltimes=[time + timedelta(hours=1) for time in alltimes_obs]
      
   else:
      alltimes=alltimes_obs

      #alltimes=alltimes_obs[0:-1]
      #alltimes_obs=alltimes_obs[0:-1]
      ## Mask times corresponding to nan values
      #alltimes_obs_masked=[]
      #alltimes_masked=[]
      #for alltime_idx in range (0,len(alltimes_obs)):
      #    if mask_obs[alltime_idx]:
      #       alltimes_obs_masked.append(alltimes_obs[alltime_idx])
      #       alltimes_masked.append(alltimes_obs[alltime_idx])
      #alltimes_obs=alltimes_obs_masked
      #alltimes=alltimes_masked

   alltimes_all=alltimes

   #######################
   ## TS PLOTS

   ######################3
   # 1B) Sea Level + Residual + TPXO signal + Seiches signal only on NOVEMBER 2022
   alltimes=alltimes_all
   plotname='sealevel_all_'+STG+post_name+'.png'
   fig=plt.figure(figsize=(16,6))
   plt.rc('font', size=18)
   fig.add_subplot(111,frame_on=False)
   #
   ax = plt.subplot(1,1,1)
   ax.xaxis.set_major_locator(mdates.DayLocator((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)))
   ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
   plt.title ('Sea Level at '+STG)

   peak_mod_tidal1 = round(np.max(mod_tidal1_sealevel[-72:]))
   peak_mod_tidal2 = round(np.max(mod_tidal2_sealevel[-72:]))
   peak_mod_tidal3 = round(np.max(mod_tidal3_sealevel[-72:]))
   peak_mod_nontidal = round(np.max(mod_nontidal_sealevel[-72:]))
   peak_obs_ismar=round(np.max(obs_ismar_sealevel[-72:]))

      # Add Extreme flood line +140 cm
   #plt.axhline(140,color='red',linewidth=1,label='Extreme floods threshold (140 cm)')
   #plt.plot(alltimes[-72:-48],mod_tidal1_sealevel[-72:-48], '-', color=colors[1],label = 'MEDFS Analysis',linewidth=3)
   #plt.plot(alltimes[-49:],mod_tidal1_sealevel[-49:], '-', color=colors[2],label = ' (max: '+str(peak_mod_tidal1)+' cm)',linewidth=3)
   plt.plot(alltimes_obs[-len(obs_ismar_sealevel):],obs_ismar_sealevel, '-o', color='navy',label = 'OBS (max: '+str(peak_obs_ismar)+' cm)',linewidth=3)
   #plt.plot(alltimes[-72:],mod_tidal2_sealevel[-72:], '-', color='tab:blue',label = 'EAS9 (max: '+str(peak_mod_tidal2)+' cm)',linewidth=3)
   #plt.plot(alltimes[-72:],mod_nontidal_sealevel[-72:], '-', color='tab:orange',label = 'EAS9-NT+TPXO (max: '+str(peak_mod_nontidal)+' cm)',linewidth=3)
   #plt.plot(alltimes[-72:],mod_tidal3_sealevel[-72:], '-', color='tab:green',label = 'EAS9-simu (max: '+str(peak_mod_tidal3)+' cm)',linewidth=3)

   plt.legend( loc='upper right',fontsize =12)
   plt.grid ()
   plt.ylabel ('Sea Level [cm]')
   #plt.xticks (color="none")
   plt.xlim([datetime(2022,11,1,0,0,0),datetime(2022,11,30,23,30,0)])
   plt.ylim([-50,200])
   #
#   ax = plt.subplot(4,1,2)
#   ax.xaxis.set_major_locator(mdates.DayLocator((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)))
#   ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
#   plt.title ('Surge')
#
#   #plt.plot(alltimes_obs,obs_ismar_fft_res_all, '-', color=colors[0],label = 'OBS (max: '+str(surge_peak_obs_ismar)+' cm)',linewidth=3)
#   #plt.plot(alltimes,mod_nontidal2_res, '-', color='lime',label = 'FC/AN EAS4 Deseiched',linewidth=3)
#   #plt.plot(alltimes,mod_nontidal_res, '-', color='orange',label = 'EAS5 Deseiched',linewidth=3)
#   plt.plot(alltimes[:-48],mod_tidal1_fft_res_all[:-48], '-', color=colors[1],label = 'EAS6 Analysis',linewidth=3)
#   #plt.plot(alltimes[-49:],mod_tidal1_fft_res_all[-49:], '-', color=colors[2],label = 'EAS6 1st day fcst (max: '+str(surge_peak_mod_tidal1)+' cm)',linewidth=3)
#   #plt.plot(alltimes[-49:],mod_tidal2_fft_res_all[-49:], '-', color=colors[3],label = 'EAS6 2nd day fcst (max: '+str(surge_peak_mod_tidal2)+' cm)',linewidth=3)
#   #plt.plot(alltimes[-49:],mod_tidal3_fft_res_all[-49:], '-', color=colors[4],label = 'EAS6 3rd day fcst (max: '+str(surge_peak_mod_tidal3)+' cm)',linewidth=3)
#
#   plt.legend( loc='upper left',fontsize = 'large' )
#   plt.grid ()
#   plt.ylabel ('Surge [cm]')
#   plt.xticks (color="none")
#   plt.xlim([datetime(2022,11,1,0,0,0),datetime(2022,11,30,23,30,0)])
#   plt.ylim([0,150])
#   #
#   ax = plt.subplot(4,1,3)
#   plt.title ('Tides')
#   ax.xaxis.set_major_locator(mdates.DayLocator((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)))
#   ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
#
#   #plt.plot(alltimes,tpxo_tid, '--', color=colors[5],label = 'Tides TPXO (max: '+str(peak_tpxo)+' cm)',linewidth=3)
##   plt.plot(alltimes,obs_ismar_fft_tid,'-', color=colors[0],label = 'OBS (max: '+str(tides_peak_obs_ismar)+' cm)',linewidth=3)
##   plt.plot(alltimes,mod_tidal1_fft_tid,'-', color=colors[2],label = 'EAS6 1st day fcst (max: '+str(tides_peak_mod_tidal1)+' cm)',linewidth=3)
#   #plt.plot(alltimes,mod_tidal2_fft_tid,'-', color=colors[3],label = 'EAS6 2nd day fcst (max: '+str(tides_peak_mod_tidal2)+' cm)',linewidth=3)
#   #plt.plot(alltimes,mod_tidal3_fft_tid,'-', color=colors[4],label = 'EAS6 3rd day fcst (max: '+str(tides_peak_mod_tidal3)+' cm)',linewidth=3)
#   #plt.plot(alltimes,obs_ismar_fft_tid,'-', color='blue',label = 'OBS TIDAL SIGNAL',linewidth=3)
#   #plt.plot(alltimes,mod_tidal_fft_tid,'-', color='red',label = 'EAS6 TIDAL SIGNAL',linewidth=3)
#
#   plt.legend( loc='upper left',fontsize = 'large' )
#   plt.grid ()
#   plt.ylabel ('Tides [cm]')
#   plt.xticks (color="none")
#   plt.xlim([datetime(2022,11,1,0,0,0),datetime(2022,11,30,23,30,0)])
#   plt.ylim([-55,55])
#   #
#   ax = plt.subplot(4,1,4)
#   plt.title ('Seiches')
#   ax.xaxis.set_major_locator(mdates.DayLocator((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)))
#   ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
#
# #  plt.plot(alltimes,obs_ismar_fft_sei,'-', color=colors[0],label = 'OBS (max: '+str(tides_peak_obs_ismar)+' cm)',linewidth=3)
#   #plt.plot(alltimes,mod_nontidal2_fft_sei,'-', color='lime',label = 'FC/AN EAS4 SEICHES SIGNAL',linewidth=3)
# #  plt.plot(alltimes[:-48],mod_nontidal1_fft_sei[:-48],'-', color=colors[1],label = 'EAS6 Analysis',linewidth=3)
# #  plt.plot(alltimes[-49:],mod_nontidal1_fft_sei[-49:],'-', color=colors[2],label = 'EAS6 1st day fcst (max: '+str(tides_peak_mod_tidal1)+' cm)',linewidth=3)
# #  plt.plot(alltimes[-49:],mod_nontidal2_fft_sei[-49:],'-', color=colors[3],label = 'EAS6 2nd day fcst (max: '+str(tides_peak_mod_tidal2)+' cm)',linewidth=3)
# #  plt.plot(alltimes[-49:],mod_nontidal3_fft_sei[-49:],'-', color=colors[4],label = 'EAS6 3rd day fcst (max: '+str(tides_peak_mod_tidal3)+' cm)',linewidth=3)
#   #plt.plot(alltimes,mod_tidal_fft_sei,'-', color='red',label = 'EAS6 SEICHES SIGNAL',linewidth=3)
#
#   plt.legend( loc='upper left',fontsize = 'large' )
#   plt.grid ()
#   plt.ylabel ('Seiches [cm]')
#   plt.xlabel ('November 2022')
#   plt.xlim([datetime(2022,11,1,0,0,0),datetime(2022,11,30,23,30,0)])
#   plt.ylim([-15,15])

   plt.savefig(workdir+plotname)
   plt.clf()



   # 2) Sea Level signal only on 20-22 NOVEMBER 2022
   alltimes=alltimes[-72:]
   plotname='SSH_'+STG+post_name+'_zoom.png'
   fig = plt.figure(figsize=(13,8))
   plt.rc('font', size=16)
   fig.add_subplot(111,frame_on=False)
   #
   ax = plt.subplot(1,1,1)
   ax.xaxis.set_major_locator(mdates.DayLocator((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)))
   ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
   ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
   plt.title (' Sea Level at '+STG)

#   # Rm mean
#   mod_tidal2_sealevel=mod_tidal2_sealevel[-72:]-np.nanmean(mod_tidal2_sealevel[-72:])+np.nanmean(obs_ismar_sealevel[-72:])
#   mod_tidal3_sealevel=mod_tidal3_sealevel[-72:]-np.nanmean(mod_tidal3_sealevel[-72:])+np.nanmean(obs_ismar_sealevel[-72:])
#   mod_nontidal_sealevel=mod_nontidal_sealevel[-72:]+tpxo_tid[-72:]-np.nanmean(mod_nontidal_sealevel[-72:]+tpxo_tid[-72:])+np.nanmean(obs_ismar_sealevel[-72:])

   # Compute the peak values:
   peak_obs_ismar  = round(np.max(obs_ismar_sealevel[-72:]))
   peak_mod_tidal1 = round(np.max(mod_tidal1_sealevel[-72:]))
   peak_mod_tidal2 = round(np.max(mod_tidal2_sealevel[-72:]))
   peak_mod_tidal3 = round(np.max(mod_tidal3_sealevel[-72:]))
   peak_mod_nontidal = round(np.max(mod_nontidal_sealevel[-72:]+tpxo_tid[-72:]-np.nanmean(tpxo_tid[-72:])))
   

      # Add Extreme flood line +140 cm
   plt.axhline(140,color='red',linewidth=1,label='Extreme floods threshold (140 cm)')
   #plt.plot(alltimes[-72:-48],mod_tidal1_sealevel[-72:-48], '-', color=colors[1],label = 'MEDFS Analysis',linewidth=3)
   #plt.plot(alltimes[-49:],mod_tidal1_sealevel[-49:], '-', color=colors[2],label = ' (max: '+str(peak_mod_tidal1)+' cm)',linewidth=3)
   plt.plot(alltimes_obs[-72:],obs_ismar_sealevel[-72:], '-o', color='navy',label = 'OBS (max: '+str(peak_obs_ismar)+' cm)',linewidth=3)
   plt.plot(alltimes[-72:],mod_tidal2_sealevel[-72:], '-', color='tab:blue',label = 'EAS9 (max: '+str(peak_mod_tidal2)+' cm)',linewidth=3)
   plt.plot(alltimes[-72:],mod_nontidal_sealevel[-72:]-np.nanmean(mod_nontidal_sealevel[-72:])+tpxo_tid[-72:]-np.nanmean(tpxo_tid[-72:])+obs_ref, '-', color='tab:orange',label = 'EAS9-NT+TPXO (max: '+str(peak_mod_nontidal)+' cm)',linewidth=3)
   plt.plot(alltimes[-72:],mod_tidal3_sealevel[-72:], '-', color='tab:green',label = 'EAS9-simu (max: '+str(peak_mod_tidal3)+' cm)',linewidth=3)
   plt.plot(alltimes[-72:],tpxo_tid[-72:]-np.nanmean(tpxo_tid[-72:])+obs_ref, '--', color=colors[5],label = 'Tides TPXO',linewidth=2)
   #plt.axhline(obs_ref,color='black',label = '3 Mon Mean = '+str(round(obs_ref,1))+' cm',linewidth=3)

   plt.axvspan(alltimes[-17],alltimes[-15], color='grey', alpha=0.2)
   plt.legend( loc='upper left',ncol=1,shadow=True, fancybox=True, fontsize=16) # 'large' )
   plt.grid ()
   plt.ylabel ('Sea Level [cm]')
   plt.xlim([datetime(2022,11,20,0,0,0),datetime(2022,11,22,23,30,0)])
   plt.ylim([-50,200])
   plt.xlabel ('November 2022')

   plt.tight_layout()
   plt.savefig(workdir+plotname,format='png', dpi=300, bbox_inches='tight') #,dpi=1200)
   plt.clf()

print ('Outputs in ',workdir)


