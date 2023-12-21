# AC Goglio Sep 2022
# Script for Forecast skill score
# Load condaE virtual env

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np
from netCDF4 import Dataset
import netCDF4 as ncdf
import datetime
from datetime import datetime
import pandas as pd
import glob
from numpy import *
import warnings
from pylab import ylabel
import matplotlib.pylab as pl
warnings.filterwarnings("ignore")
mpl.use('Agg')
#####################################

# -- Workdir path -- 
workdir = '/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_plots_new/'

# Offset computed over Nov 2019
mod_mean6  = -0.0995  # Mean over Nov 2019 run EAS6_AN_w10 in ISMAR_TG
mod_mean5  = -0.0992  # Mean over Nov 2019 run EAS5_AN_w10 in ISMAR_TG
tpxo_mean  = -0.0004  # Mean over Nov 2019 tpxo in ISMAR_TG
obs_mean   = 0.7524  # Mean over Nov 2019 obs in ISMAR_TG (Manu's value: 0.67 )

# length of the time interval to be plotted: allp, zoom or super-zoom or osr5
time_p = 'allp'

# To add a post_plot_name
pp_name='_new'

# To interpolate the obs from hh:00 to hh:30 and plot mod-obs diffs put obs_interp_flag = 1
obs_interp_flag = 0

# To add tpxo tides to non-tidal runs
flag_add_tides = 1

# To use old tpxo ('Manu') 
flag_oldtpxo = 0

# ---  Input archive ---
input_dir          = '/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_sea_level/' 
tpxo_ts            = 'ISMAR_TG_tpxo.nc'
tpxo2_ts           = 'ISMAR_TG_tpxo_Manu.txt' # TMP 
tpxo3_ts           = 'ISMAR_TG_tpxo_inst.txt' # TMP 
#
input_tg   = ['ISMAR_TG']
input_dat  = ['obs','mod'] # Do not change the order because the obs are used as reference for offset and differences!
input_type = ['FC1','FC2','FC3','AN']
input_res  = ['08','08_12','10'] # Do not change the order  '08','08sub','08_12','10'
input_sys  = ['EAS5','EAS6']

input_var     = 'sossheig' 
udm           = 'm'
input_obs_timevar = 'TIME'
input_mod_timevar = 'time_counter'

# Color
#colors = pl.cm.jet_r(np.linspace(0,1,24))
colors_5 = pl.cm.Blues(np.linspace(0.2,1,12))
colors_6 = pl.cm.Greens(np.linspace(0.2,1,12))
colors = np.vstack((colors_5, colors_6))
#############################
# TMP tpxo 2: 
fh2 = pd.read_csv(input_dir+'/'+tpxo2_ts,sep=' ',comment='#',header=None)
var_tpxo2 = fh2[4][:]
var_tpxo2 = np.array(var_tpxo2)

fh3 = pd.read_csv(input_dir+'/'+tpxo3_ts,sep=' ',comment='#',header=None)
var_tpxo3 = fh3[4][:]
var_tpxo3 = np.array(var_tpxo3)

######

# Loop on tide-gauges
for tg_idx,tg in enumerate(input_tg):

    # Output file
    fig_name = workdir+'/'+tg+'_'+time_p+pp_name+'.png'

    # Loop on datasets
    for dat_idx,dat in enumerate(input_dat):

        # OBS
        if dat == 'obs':
           # input files name
           file_to_open = input_dir+'/'+tg+'_'+dat+'.csv' #+'.nc'+'_00.csv'
           file_to_open_long = input_dir+'/'+tg+'_'+dat+'_long.csv'
           print ('Open files: ',file_to_open,file_to_open_long)
           # check the existence of the file and open it
           if glob.glob(file_to_open):
              #fh = ncdf.Dataset(file_to_open,mode='r')
              fh = pd.read_csv(file_to_open,sep=';',comment='#',header=None)
              # Read time axes and compute time-var
              #time_obs   = fh.variables[input_obs_timevar][:]
              #time_obs_units = fh.variables[input_obs_timevar].getncattr('units')
              #alltimes_obs=[]
              #for alltime_idx in range (0,len(time_obs)):
              #    alltimes_obs.append(datetime(ncdf.num2date(time_obs[alltime_idx],time_obs_units).year,ncdf.num2date(time_obs[alltime_idx],time_obs_units).month,ncdf.num2date(time_obs[alltime_idx],time_obs_units).day,ncdf.num2date(time_obs[alltime_idx],time_obs_units).hour,ncdf.num2date(time_obs[alltime_idx],time_obs_units).minute,ncdf.num2date(time_obs[alltime_idx],time_obs_units).second))
              alltimes_obs = fh[0][:]

              # Read obs time series
              #var_obs  = fh.variables[input_var][:]
              var_obs = fh[1][:] #*100.0
              var_obs = np.array(var_obs)
              # Interpolate from :00 to :30
              if obs_interp_flag == 1:
                 where_to_interp = np.linspace(0.5,float(len(var_obs))+0.5,216)
                 interp_obs = np.interp(linspace(0.5,len(var_obs)+0.5,216),range(0,len(var_obs)),var_obs)
                 var_obs = interp_obs

              # Currently the values to compute the offset are obtained externally and passed as vars
              ## Read the mean of the obs (to compute the offset)
              #if glob.glob(file_to_open_long):
              #   fh_long = pd.read_csv(file_to_open_long,sep=';',comment='#',header=None)
              #   var_obs_long = fh[1][:]
              #   var_obs_long = np.array(var_obs_long)
              #   if obs_interp_flag == 1:
              #      # Interpolate from :00 to :30
              #      where_to_interp_long = np.linspace(0.5,float(len(var_obs))+0.5,216)
              #      interp_obs_long = np.interp(linspace(0.5,len(var_obs_long)+0.5,216),range(0,len(var_obs_long)),var_obs_long)
              #      var_obs_long = interp_obs_long
              #
              # Compute the long mean of the obs
              #try:
              #   obs_mean = np.nanmean(var_obs_long)
              #   print ('Obs mean:',obs_mean)
              #   mean_obs = np.nanmean(var_obs)
              #except:
              #   mean_obs = np.nanmean(var_obs)
              #   obs_mean = mean_obs

              offset6 = obs_mean-mod_mean6
              offset5 = obs_mean-mod_mean5
              print ('Offsets 6/5',offset6,offset5)

              # Close infile 
              #fh.close()
           else:
              print ('NOT Found!')  
        # MOD
        elif dat == 'mod':
            # Loop on system versions
            for easys_idx,easys in enumerate(input_sys):
                # Loop on ts type
                for atype_idx,atype in enumerate(input_type):
                    # Loop on atm forcing spatial resolutions
                    for res_idx,res in enumerate(input_res):
                        # input file name
                        #infile = globals()[tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res+'.nc']
                        file_to_open = input_dir+'/'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res+'.nc' #_00.nc_only.nc
                        print ('Open file: ',file_to_open)
                        # build the arrays to sore the differences wrt obs dataset
                        globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                        # check the existence of the file and open it
                        print ('Try: ',file_to_open+'_ok.nc') #_00.nc_only.nc
                        if glob.glob(file_to_open+'_ok.nc'): #_00.nc_only.nc
                           fh = ncdf.Dataset(file_to_open+'_ok.nc',mode='r') #_00.nc_only.nc
                           # Read time axes and compute time-var
                           time_mod   = fh.variables[input_mod_timevar][:]
                           time_mod_units = fh.variables[input_mod_timevar].getncattr('units')
                           globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                           for alltime_idx in range (0,len(time_mod)):
                               globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res].append(datetime(ncdf.num2date(time_mod[alltime_idx],time_mod_units).year,ncdf.num2date(time_mod[alltime_idx],time_mod_units).month,ncdf.num2date(time_mod[alltime_idx],time_mod_units).day,ncdf.num2date(time_mod[alltime_idx],time_mod_units).hour,ncdf.num2date(time_mod[alltime_idx],time_mod_units).minute,ncdf.num2date(time_mod[alltime_idx],time_mod_units).second))
                           # Read mod time series
                           fh = ncdf.Dataset(file_to_open+'_ok.nc',mode='r')
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = fh.variables[input_var][:] 


                           # Mv to the obs mean 
                           #offset6 = np.nanmean(var_obs)-np.nanmean(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]) # TMP 2 be rm
                           #offset5 = offset6 # TMP 2 be rm
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+offset6
                           #globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+offset6-np.nanmean(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                           # if EAS5 add tpxo tides and mv to obs mean
                           if easys == 'EAS5' and flag_add_tides == 1:
                              fh = ncdf.Dataset(input_dir+tpxo_ts,mode='r')
                              tpxo_sig = fh.variables['tide_z'][:]
                              tpxo_diffs = np.squeeze(tpxo_sig)-var_tpxo2 # TMP
                              globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+tpxo_sig+offset5-offset6-tpxo_mean
                              #globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+tpxo_sig+offset5-np.nanmean(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+tpxo_sig)

                           # Compute the differences wrt obs
                           globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])-var_obs
                           # Close infile
                           fh.close()
                        elif glob.glob(file_to_open):
                           print ('All time steps in the original file!')
                           fh = ncdf.Dataset(file_to_open,mode='r')
                           # Read time axes and compute time-var
                           time_mod   = fh.variables[input_mod_timevar][:]
                           time_mod_units = fh.variables[input_mod_timevar].getncattr('units')
                           globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                           for alltime_idx in range (0,len(time_mod)):
                               globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res].append(datetime(ncdf.num2date(time_mod[alltime_idx],time_mod_units).year,ncdf.num2date(time_mod[alltime_idx],time_mod_units).month,ncdf.num2date(time_mod[alltime_idx],time_mod_units).day,ncdf.num2date(time_mod[alltime_idx],time_mod_units).hour,ncdf.num2date(time_mod[alltime_idx],time_mod_units).minute,ncdf.num2date(time_mod[alltime_idx],time_mod_units).second))
                           # Read mod time series
                           fh = ncdf.Dataset(file_to_open,mode='r')
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = fh.variables[input_var][:]
                           # Mv to the obs mean 
                           #offset6 = np.nanmean(var_obs)-np.nanmean(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]) # TMP 2 be rm
                           #offset5 = offset6 # TMP 2 be rm

                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+offset6
                           #globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+offset6-np.nanmean(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                           # if EAS5 add tpxo tides and mv to obs mean
                           if easys == 'EAS5' and flag_add_tides == 1:
                              fh = ncdf.Dataset(input_dir+tpxo_ts,mode='r')
                              tpxo_sig = fh.variables['tide_z'][:]
                              tpxo_diffs = np.squeeze(tpxo_sig)-var_tpxo2 # TMP
                              globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+tpxo_sig+offset5-offset6-tpxo_mean
                              #globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+tpxo_sig+offset5-np.nanmean(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+tpxo_sig)

                           # Close infile
                           fh.close()

                           ## Read TPXO and Tpxo Manu TMP
                           #fh = ncdf.Dataset(input_dir+tpxo_ts,mode='r')
                           #tpxo_sig = fh.variables['tide_z'][:]
                           #fh.close()
                           ##
                           #fh2 = pd.read_csv(input_dir+'/'+tpxo2_ts,sep=' ',comment='#',header=None)
                           #var_tpxo2 = fh2[4][:]
                           #var_tpxo2 = np.array(var_tpxo2)

                           # Compute the differences wrt obs
                           globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])-var_obs
                        else:
                           print ('NOT Found!')


######## PLOT TS #########
# Loop on tide-gauges
for tg_idx,tg in enumerate(input_tg):
    # Initialize the plot
    fig = plt.figure(0,figsize=(20,11))
    plt.rc('font', size=16)
    print ('Plot: ',fig_name)
    print ('Name, max peak 12 Nov 2019, max peak 15 Nov 2019')
    if obs_interp_flag == 1:
       fig.add_subplot(111)
       gs = fig.add_gridspec(2, 3)
       # Abs values
       ax = plt.subplot(gs[0, :-1]) #(2,2,1)
    else:
       ax = fig.add_subplot(111)
    # Line index
    idx_line_plot = 0
    # Loop on datasets
    for dat_idx,dat in enumerate(input_dat):
        # MOD
        if dat == 'mod':
            # Loop on system versions
            for easys_idx,easys in enumerate(input_sys):
                # Loop on ts type
                for atype_idx,atype in enumerate(input_type):
                    # Loop on atm forcing resolutions
                    for res_idx,res in enumerate(input_res):

                        #if atype == 'FC3': # TMPissimo
                        #if atype == 'FC1' or atype == 'FC2' or atype == 'FC3': #TMPissimo
                        #if easys == 'EAS6':
                        #if easys == 'EAS5' and (atype == 'FC1' or atype == 'FC2' or atype == 'FC3') : #TMPissimo
                        #if res == '08_12' : #TMPissimo

                          # Plot the mod lines in the plot (line type based on spatial res)
                          # if res == '08' :
                          #    ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                          # elif res == '10' :
                          #    ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),':',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                          # elif res == '08sub' :
                          #    if atype != 'AN':
                          #       ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'--',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)

                           # Plot the mod lines in the plot (line type based on time res)
                           try: 
                              time4plot = np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                              line4plot = np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                              # TMP 
                              if easys == 'EAS5' and flag_oldtpxo == 1:
                                 line4plot = line4plot - tpxo_diffs
                           except:
                              print (' ')

                           if idx_line_plot == 0:
                              # OBS
                              ax.plot(time4plot,var_obs*100,'-',color='red',label='OBS',linewidth=3)
                              print ('Obs',int(np.max(var_obs)*100),int(np.max(var_obs[96:120])*100))

                           if res == '10' and (atype == 'FC1' or atype == 'FC2' or atype == 'FC3') :
                              ax.plot(time4plot,line4plot*100,':',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=3)
                              #ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),':',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                           elif (res == '08' or res == '08_12') and (atype == 'FC1' or atype == 'FC2' or atype == 'FC3') :
                                 ax.plot(time4plot,line4plot*100,'--',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=3)
                                 #ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'--',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                           else:
                              try:
                                 ax.plot(time4plot,line4plot*100,color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=3)
                                 #ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                              except:
                                 print (' ')

                           # Update line in plot index
                           idx_line_plot = idx_line_plot + 1

                           # print the max for peak 1
                           name2print=easys+'_'+atype+'_w'+res
                           max2print=int(np.max(line4plot[43:48]*100))
                           # print the max for peak 2
                           max2print_15=int(np.max(line4plot[96:120]*100))
                           print (name2print,max2print,max2print_15)               
                        ## Update line in plot index
                        #idx_line_plot = idx_line_plot + 1

    # OBS
    #ax.plot(time4plot,var_obs*100,'-',color='red',label='OBS',linewidth=3)
    #print ('Obs',int(np.max(var_obs)*100),int(np.max(var_obs[96:120])*100))
    #
    # TPXO
    #if flag_oldtpxo != 1:
    #   ax.plot(time4plot,(np.squeeze(tpxo_sig)+obs_mean)*100,'--',color='black',label='Tides TPXO',linewidth=2)
    #else:
    #   print ('old tpxo')
    #   ax.plot(time4plot,(np.squeeze(tpxo_sig)-tpxo_diffs+obs_mean)*100,'--',color='black',label='Tides TPXO',linewidth=2)
    ##ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),var_obs,'o-',color='black',label='OBS',linewidth=1.5)
    ##ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(tpxo_sig)+obs_mean,'--',color='black',label='TPXO',linewidth=1.5)
    ##ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(var_tpxo2)+obs_mean,'--',color='magenta',label='TPXO old',linewidth=1.5)

    # Add Extreme flood line +140 cm 
    plt.axhline(140,color='black',linewidth=2)

    # Finalize the plot
    ylabel("Sea Level [cm]",fontsize=18)
    box = ax.get_position()
    ax.plot(time4plot, np.zeros(len(time4plot)), color='w', alpha=0, label='  ')
    #leg = plt.legend(loc='lower right',ncol=3,  shadow=True, fancybox=True, fontsize=15)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1, fancybox=True, shadow=True)
    ax.grid('on')
    #plt.axhline(linewidth=2, color='black')
    plt.title('Sea Level time-series in '+tg,fontsize=18)
    plt.ylim(0,200)
    if time_p == 'allp' :
       plt.xlim([datetime(2019,11,11,0,0,0),datetime(2019,11,15,23,30,0)])
       plt.xlabel ('November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       ax.margins(x=0)
    elif time_p == 'zoom' :
       plt.xlim([datetime(2019,11,12,0,0,0),datetime(2019,11,13,23,30,0)])
       plt.xlabel ('November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       #ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%H"))
       ax.margins(x=0)
    elif time_p == 'superzoom' :
       plt.xlim([datetime(2019,11,12,16,30,0),datetime(2019,11,12,23,30,0)])
       #plt.xlabel ('12 November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.HourLocator())
       #ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%H"))
       ax.margins(x=0)
    elif time_p == 'osr5' :
       plt.xlim([datetime(2019,11,11,0,0,0),datetime(2019,11,15,23,30,0)])
       plt.xlabel ('November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       ax.margins(x=0)
       plt.ylim(0,2)



    # Diffs PLOT
    if obs_interp_flag == 1:
      # Line index
      idx_line_plot = 0
      ax = plt.subplot(gs[1, :-1]) #(2,2,3)
      # Loop on datasets
      for dat_idx,dat in enumerate(input_dat):
        # MOD
        if dat == 'mod':
            # Loop on system versions
            for easys_idx,easys in enumerate(input_sys):
                # Loop on ts type
                for atype_idx,atype in enumerate(input_type):
                    # Loop on atm forcing resolutions
                    for res_idx,res in enumerate(input_res):
 
                        #if atype == 'AN': # TMPissimo
                        #if atype == 'FC3': # TMPissimo
                        #if easys == 'EAS6' : #TMPissimo
                        #if easys == 'EAS5' and (atype == 'FC1' or atype == 'FC2' or atype == 'FC3') : #TMPissimo

                           # Plot the mod lines in the plot (line type based on spatial res) and compiute the differences wrt obs 
                           #if res == '08' :
                           #   try: 
                           #     ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res+' ('+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+','+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2))+')',linewidth=1.5)
                           #     print (easys+'_'+atype+'_w'+res+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2)))
                           #   except:
                           #     ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5) 
                           #     print (easys+'_'+atype+'_w'+res+' ')
                           #elif res == '10' :
                           #   try:
                           #      ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),':',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res+' ('+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+','+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2))+')',linewidth=1.5)
                           #      print (easys+'_'+atype+'_w'+res+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2)))
                           #   except:
                           #      ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),':',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                           #      print (easys+'_'+atype+'_w'+res+' ')
                           #elif res == '08sub' :
                           #   if atype != 'AN':
                           #      try:
                           #         ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'--',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res+' ('+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+','+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2))+')',linewidth=1.5)
                           #         print (easys+'_'+atype+'_w'+res+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2)))
                           #      except:
                           #         ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'--',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                           #         print (easys+'_'+atype+'_w'+res+' ')

                           # Plot the mod lines in the plot (line type based on time res) and compiute the differences wrt obs 
                           if res == '10' and (atype == 'FC1' or atype == 'FC2' or atype == 'FC3'):
                              try:
                                 ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),':',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res+' ('+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+','+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2))+')',linewidth=1.5)
                                 print (easys+'_'+atype+'_w'+res+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2)))
                              except:
                                 ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),':',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                                 print (easys+'_'+atype+'_w'+res+' ')
                           elif res == '08' and (atype == 'FC1' or atype == 'FC2' or atype == 'FC3') :
                                 try:
                                    ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'--',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res+' ('+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+','+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2))+')',linewidth=1.5)
                                    print (easys+'_'+atype+'_w'+res+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2)))
                                 except:
                                    ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'--',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                                    print (easys+'_'+atype+'_w'+res+' ')
                           else:
                              try: 
                                ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res+' ('+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+','+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2))+')',linewidth=1.5)
                                print (easys+'_'+atype+'_w'+res+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][92],2))+' '+str(round(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][93],2)))
                              except:
                                try: 
                                   ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
                                except: 
                                   print (' ')
                                #print (easys+'_'+atype+'_w'+res+' ')
  
                           # Update line in plot index
                           idx_line_plot = idx_line_plot + 1
                        ## Update line in plot index
                        #idx_line_plot = idx_line_plot + 1

      # Finalize the plot
      plt.axhline(linewidth=2, color='black')
      ylabel("SSH diff [m]",fontsize=18)
      plt.xlabel ('November 2019',fontsize=18)
      #box = ax.get_position()
      #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
      leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2,  shadow=True, fancybox=True, fontsize=12)
      #leg.get_frame().set_alpha(0.3)
      ax.grid('on')
      #plt.axhline(linewidth=2, color='black')
      #plt.title('SSH time-series in '+tg,fontsize=18)
      if time_p == 'allp' :
       plt.xlim([datetime(2019,11,11,0,0,0),datetime(2019,11,15,23,30,0)])
       plt.xlabel ('November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       ax.margins(x=0)
      elif time_p == 'zoom' :
       plt.xlim([datetime(2019,11,12,0,0,0),datetime(2019,11,13,23,30,0)])
       plt.xlabel ('November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       #ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%H"))
       ax.margins(x=0)
      elif time_p == 'superzoom' :
       plt.xlim([datetime(2019,11,12,16,30,0),datetime(2019,11,12,23,30,0)])
       plt.xlabel ('12 November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.HourLocator())
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%H"))
       ax.margins(x=0)
      elif time_p == 'osr5' :
       plt.xlim([datetime(2019,11,11,0,0,0),datetime(2019,11,15,23,30,0)])
       plt.xlabel ('November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       ax.margins(x=0)

    plt.tight_layout()
    plt.savefig(fig_name,format='png',dpi=1200)
    print ('Done!')
    plt.clf()

### TPXO diffs plot TMP
#fig = plt.figure(0,figsize=(16,8))
#plt.rc('font', size=16)
#plt.title('SSH time-series TPXO diff '+tg,fontsize=18)
#plt.plot(time4plot,np.squeeze(tpxo_sig),'-',label='TPXO h mean',linewidth=1.5)
#plt.plot(time4plot,np.squeeze(var_tpxo2),'-',label='TPXO old',linewidth=1.5)
#plt.plot(time4plot,np.squeeze(var_tpxo3),'-',label='TPXO inst',linewidth=1.5)
##a.plot(arange(0,len(tpxo_diffs)),tpxo_diffs,'-',color='black',label='TPXO - TPXO old',linewidth=1.5)
##plt.plot(time4plot,np.squeeze(tpxo_sig)-np.squeeze(var_tpxo3))
#plt.grid()
#plt.xlabel ('November 2019')
#plt.ylabel ('Tides [m]')
#print ('Prove',var_tpxo2,var_tpxo3,len(var_tpxo2),len(var_tpxo3))
#leg = plt.legend(loc='lower right', ncol=1,  shadow=True, fancybox=True, fontsize=15)
##
#plt.savefig(workdir+'/tpxodiffs.png',format='png',dpi=1200)
#print ('Done!')
#plt.clf()
  

