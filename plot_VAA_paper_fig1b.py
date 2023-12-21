# AC Goglio Sep 2022
# Script for Forecast skill score
# Load condaE virtual env!

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
workdir = '/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_plots_new2//'

# -- Period --
start_date = 20191109 #12 #09
end_date   = 20191115 #13 #17

mod_mean6  = -0.0995  # Mean over Nov 2019 run EAS6_AN_w10 in ISMAR_TG
mod_mean5  = -0.0992  # Mean over Nov 2019 run EAS5_AN_w10 in ISMAR_TG
tpxo_mean  = -0.0004  # Mean over Nov 2019 tpxo in ISMAR_TG
#obs_mean   =  0.7524  # Mean over Nov 2019 obs in ISMAR_TG
obs_mean   =  0.76  # Mean over 10-15 Nov 2019 obs in ISMAR_TG

# length of the time interval to be plotted: allp, zoom or super-zoom
time_p = 'osr5'

# To interpolate the obs from hh:00 to hh:30 and plot mod-obs diffs
obs_interp_flag = 1

# ---  Input archive ---
input_dir          = '/work/oda/med_dev//Venezia_Acqua_Alta_2019/VAA_sea_level_paper/'
tpxo_ts            = 'ISMAR_TG_tpxo.nc'
#
input_tg   = ['ISMAR_TG']
input_dat  = ['obs','mod'] # Do not change the order because the obs are used as reference for offset and differences!
input_type = ['FCall_20191110','FCall_20191111','FCall_20191112'] #['AN','FCall_20191109','FCall_20191110','FCall_20191111','FCall_20191112','FCall_20191113','FCall_20191114','FCall_20191115'] # Leadtime of the forecasts
input_res  = ['10'] #['08','08sub','10'] # Do not change the order 
input_sys  = ['EAS6']

input_var     = 'sossheig' 
udm           = 'm'
input_obs_timevar = 'TIME'
input_mod_timevar = 'time_counter'

# Color
colors = ['darkblue','tab:blue','tab:cyan'] 
#colors = pl.cm.jet_r(np.linspace(0,1,16))

#############################
# Loop on tide-gauges
for tg_idx,tg in enumerate(input_tg):

    # Output file
    fig_name = workdir+'/'+tg+'_FCall_'+time_p+'.png' #'zoom.png'

    # Loop on datasets
    for dat_idx,dat in enumerate(input_dat):

        # OBS
        if dat == 'obs':
           # input files name
           file_to_open = input_dir+'/'+tg+'_'+dat+'_10-12.csv' #+'.nc'
           file_to_open_long = input_dir+'/'+tg+'_'+dat+'_long.csv'
           file_to_open_Hfreq = input_dir+'/'+tg+'_'+dat+'_10min.csv_ok.csv'
           print ('Open files: ',file_to_open,file_to_open_long,file_to_open_Hfreq)
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
              if obs_interp_flag == 2:
                 where_to_interp = np.linspace(0.5,float(len(var_obs))+0.5,len(var_obs))
                 interp_obs = np.interp(linspace(0.5,len(var_obs)+0.5,len(var_obs)),range(0,len(var_obs)),var_obs)
                 var_obs = interp_obs

              # TMP:
              var_obs = var_obs[1:]
              print ('Prova',var_obs)

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
              print ('NOT Found!',file_to_open)  

           # Read high freq obs
           if glob.glob(file_to_open_Hfreq):
              fh_Hfreq = pd.read_csv(file_to_open_Hfreq,sep=';',comment='#',header=None)
              # Read time axes and compute time-var
              #time_obs   = fh.variables[input_obs_timevar][:]
              #time_obs_units = fh.variables[input_obs_timevar].getncattr('units')
              #alltimes_obs=[]
              #for alltime_idx in range (0,len(time_obs)):
              #    alltimes_obs.append(datetime(ncdf.num2date(time_obs[alltime_idx],time_obs_units).year,ncdf.num2date(time_obs[alltime_idx],time_obs_units).month,ncdf.num2date(time_obs[alltime_idx],time_obs_units).day,ncdf.num2date(time_obs[alltime_idx],time_obs_units).hour,ncdf.num2date(time_obs[alltime_idx],time_obs_units).minute,ncdf.num2date(time_obs[alltime_idx],time_obs_units).second))
              alltimes_obs = fh_Hfreq[0][:]

              # Read obs time series
              #var_obs  = fh.variables[input_var][:]
              var_obs_Hfreq = fh_Hfreq[1][:] #*100.0
              var_obs_Hfreq = np.array(var_obs_Hfreq)
              # Interpolate from :00 to :30
              #if obs_interp_flag == 1:
              #   where_to_interp = np.linspace(0.5,float(len(var_obs))+0.5,len(var_obs))
              #   interp_obs = np.interp(linspace(0.5,len(var_obs)+0.5,len(var_obs)),range(0,len(var_obs)),var_obs)
              #   var_obs = interp_obs
              # TMP:
              var_obs_Hfreq = var_obs_Hfreq[0:144*6]


 
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
                        file_to_open = input_dir+'/'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res+'.nc'
                        print ('-----')
                        # build the arrays to sore the differences wrt obs dataset
                        #globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                        # check the existence of the file and open it
                        if glob.glob(file_to_open+'_ok.nc'):
                           print ('Open file: ',file_to_open+'_ok.nc')
                           fh = ncdf.Dataset(file_to_open+'_ok.nc',mode='r')
                           # Read time axes and compute time-var
                           time_mod   = fh.variables[input_mod_timevar][:]
                           time_mod_units = fh.variables[input_mod_timevar].getncattr('units')
                           globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                           globals()['alltimes_mod_Hfreq_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                           if atype == 'FCall_20191110':
                              for alltime_idx in range (0,len(time_mod)):
                                  globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res].append(datetime(ncdf.num2date(time_mod[alltime_idx],time_mod_units).year,ncdf.num2date(time_mod[alltime_idx],time_mod_units).month,ncdf.num2date(time_mod[alltime_idx],time_mod_units).day,ncdf.num2date(time_mod[alltime_idx],time_mod_units).hour,ncdf.num2date(time_mod[alltime_idx],time_mod_units).minute,ncdf.num2date(time_mod[alltime_idx],time_mod_units).second))
                                  for hfreq_idx in range(0,60,10):
                                      globals()['alltimes_mod_Hfreq_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res].append(datetime(ncdf.num2date(time_mod[alltime_idx],time_mod_units).year,ncdf.num2date(time_mod[alltime_idx],time_mod_units).month,ncdf.num2date(time_mod[alltime_idx],time_mod_units).day,ncdf.num2date(time_mod[alltime_idx],time_mod_units).hour,ncdf.num2date(hfreq_idx*60,time_mod_units).minute,ncdf.num2date(time_mod[alltime_idx],time_mod_units).second))
                           # Read mod time series
                           fh = ncdf.Dataset(file_to_open+'_ok.nc',mode='r')
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = fh.variables[input_var][:] 


                           # Mv to the obs mean 
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+offset6
                           # if EAS5 add tpxo tides and mv to obs mean
                           if easys == 'EAS5':
                              fh = ncdf.Dataset(input_dir+tpxo_ts,mode='r')
                              tpxo_sig = fh.variables['tide_z'][:]
                              globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+tpxo_sig+offset5-offset6-tpxo_mean

                           # Compute the differences wrt obs
                           #globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])-var_obs # 
                           # Close infile
                           fh.close()
                        elif glob.glob(file_to_open):
                           #print ('All time steps in the original file!')
                           print ('Open file: ',file_to_open)
                           fh = ncdf.Dataset(file_to_open,mode='r')
                           # Read time axes and compute time-var
                           time_mod   = fh.variables[input_mod_timevar][:]
                           time_mod_units = fh.variables[input_mod_timevar].getncattr('units')
                           globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                           globals()['alltimes_obs_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                           globals()['alltimes_mod_Hfreq_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                           print ('P1 ',atype)
                           if atype == 'FCall_20191110' or atype == 'FCall_20191111' or atype == 'FCall_20191112':
                              print ('P2')
                              for alltime_idx in range (0,len(time_mod)):
                                  globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res].append(datetime(ncdf.num2date(time_mod[alltime_idx],time_mod_units).year,ncdf.num2date(time_mod[alltime_idx],time_mod_units).month,ncdf.num2date(time_mod[alltime_idx],time_mod_units).day,ncdf.num2date(time_mod[alltime_idx],time_mod_units).hour,ncdf.num2date(time_mod[alltime_idx],time_mod_units).minute,ncdf.num2date(time_mod[alltime_idx],time_mod_units).second))
                                  globals()['alltimes_obs_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res].append(datetime(ncdf.num2date(time_mod[alltime_idx],time_mod_units).year,ncdf.num2date(time_mod[alltime_idx],time_mod_units).month,ncdf.num2date(time_mod[alltime_idx],time_mod_units).day,ncdf.num2date(time_mod[alltime_idx],time_mod_units).hour,ncdf.num2date(0,time_mod_units).minute,ncdf.num2date(time_mod[alltime_idx],time_mod_units).second))
                                  for hfreq_idx in range(0,60,10):
                                      globals()['alltimes_mod_Hfreq_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res].append(datetime(ncdf.num2date(time_mod[alltime_idx],time_mod_units).year,ncdf.num2date(time_mod[alltime_idx],time_mod_units).month,ncdf.num2date(time_mod[alltime_idx],time_mod_units).day,ncdf.num2date(time_mod[alltime_idx],time_mod_units).hour,ncdf.num2date(hfreq_idx*60,time_mod_units).minute,ncdf.num2date(time_mod[alltime_idx],time_mod_units).second))
                
                           # tpxo tides
                           fh = ncdf.Dataset(input_dir+tpxo_ts,mode='r')
                           tpxo_sig = fh.variables['tide_z'][:]
                           fh.close()
                           # Read mod time series
                           fh = ncdf.Dataset(file_to_open,mode='r')
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = fh.variables[input_var][:]

                           # Mv to the obs mean 
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+offset6
                           # if EAS5 add tpxo tides and mv to obs mean
                           if easys == 'EAS5':
                              fh = ncdf.Dataset(input_dir+tpxo_ts,mode='r')
                              tpxo_sig = fh.variables['tide_z'][:]
                              globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]+tpxo_sig+offset5-offset6-tpxo_mean

                           # Close infile
                           fh.close()
                           print ('Time mod',globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                           # Compute the differences wrt obs
                           #globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])-var_obs
                        else:
                           print ('NOT Found!',file_to_open)

#print ('Time mod',globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
#print ('Time_obs',alltimes_obs)
######## PLOT TS #########
# Loop on tide-gauges
for tg_idx,tg in enumerate(input_tg):
    # Initialize the plot
    print ('Plot: ',fig_name)
    fig = plt.figure(figsize=(13,8))
    plt.rc('font', size=16)
    ax = fig.add_subplot(111)
    #gs = fig.add_gridspec(2, 3)
    #ax = plt.subplot(gs[0, :-1])
    # Line index
    idx_line_plot = 0
    # Loop on datasets
    for dat_idx,dat in enumerate(input_dat):
        # MOD
        if dat == 'mod':
            # Loop on system versions
            for easys_idx,easys in enumerate(input_sys):
                # Loop on atm forcing resolutions
                for res_idx,res in enumerate(input_res):
                    # Loop on ts type
                    for atype_idx,atype in enumerate(input_type):

                           # print the max for peak 1
                           if atype == 'FCall_20191110':
                              max2print=int(np.max(np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[72-6:72]*100))
                              obs_max=int(np.max(var_obs[72-6:72])*100)
                              obs_Hfreq_max=int(np.max(var_obs_Hfreq[(72-6)*6:72*6])*100)
                              diff2print=max2print-obs_max
                              lab2print='Forecast 20191110'
                           elif atype == 'FCall_20191111':
                              max2print=int(np.max(np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[(72-6)-24:72-24]*100))
                              obs_max=int(np.max(var_obs[72-6:72])*100)
                              obs_Hfreq_max=int(np.max(var_obs_Hfreq[(72-6)*6:72*6])*100)
                              diff2print=max2print-obs_max
                              lab2print='Forecast 20191111'
                           elif atype == 'FCall_20191112':
                              max2print=int(np.max(np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[(72-6)-48:72-48]*100))
                              obs_max=int(np.max(var_obs[72-6:72])*100)
                              obs_Hfreq_max=int(np.max(var_obs_Hfreq[(72-6)*6:72*6])*100)
                              diff2print=max2print-obs_max
                              lab2print='Forecast 20191112' 

                           # Plot the mod lines in the plot (line type based on time res)
                           if easys == 'EAS5' and atype != 'AN' :
                              ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])*100,'-',color=colors[idx_line_plot],label=lab2print+' (12 November peak ='+str(max2print)+' cm)',linewidth=3)
                           elif easys == 'EAS6' and atype != 'AN' and res != '08' :
                                 ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])*100,'-',color=colors[idx_line_plot],label=lab2print+' (12 November peak = '+str(max2print)+' cm)',linewidth=3,zorder=3)
                           elif res != '08' :
                              try:
                                 ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])*100,'-',color=colors[idx_line_plot],label=lab2print+' (12 November peak = '+str(max2print)+' cm)',linewidth=3)
                              except:
                                 print ('No')

                           # Update line in plot index
                           idx_line_plot = idx_line_plot + 1

    # High freq OBS
    obs_max=int(np.max(var_obs[68:72])*100)
    obs_H_mean=round(np.nanmean(var_obs[0:144])*100,2)
    obs_Hfreq_max=int(np.max(var_obs_Hfreq[68*6:72*6])*100)
    obs_Hfreq_mean=round(np.nanmean(var_obs_Hfreq[0:144*6])*100,2)
    print ('obs_Hfreq_max',obs_Hfreq_max)
    print ('obs_H_mean,obs_Hfreq_mean',obs_H_mean,obs_Hfreq_mean)
    # HF OBS
    ax.plot(np.squeeze(globals()['alltimes_mod_Hfreq_'+tg+'_'+dat+'_'+easys+'_'+'FCall_20191110'+'_w'+res])[:-6],var_obs_Hfreq[6:(24*3*6)]*100,'o-',color='orange',label='High Freq. OBS (12 November peak = '+str(obs_Hfreq_max)+' cm)',linewidth=3,zorder=1)

    # OBS
    #ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[23+1:],var_obs[23:-1]*100,'o-',color='red',label='Hourly OBS (max='+str(obs_max)+' cm)',linewidth=3,zorder=1)
    ax.plot(np.squeeze(globals()['alltimes_obs_'+tg+'_'+dat+'_'+easys+'_'+'FCall_20191110'+'_w'+res])[:],var_obs[:(24*3)]*100,'o-',color='red',label='Hourly OBS (12 November peak = '+str(obs_max)+' cm)',linewidth=3,zorder=2)
    # Add mean obs offset
    plt.axhline(obs_mean*100,color='red',linewidth=2,linestyle='dashed',label='Mean OBS',zorder=0)


    # TPXO
    #ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),(np.squeeze(tpxo_sig)-tpxo_mean+obs_mean)*100,'--',color='black',label='Tides TPXO',linewidth=2)

    # Add Extreme flood line +140 cm 
    plt.axhline(140,color='black',linewidth=2)

    # Finalize the plot
    ylabel("Sea Level [cm]",fontsize=18)
    box = ax.get_position()
    #ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]), np.zeros(len(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]))), color='w', alpha=0, label='  ')
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,  shadow=True, fancybox=True)
    ax.leg = plt.legend(loc='upper left', ncol=1,  shadow=True, fancybox=True)
    ##leg = plt.legend(loc='lower right', ncol=2,  shadow=True, fancybox=True, fontsize=12)
    ##leg.get_frame().set_alpha(0.3)
    ax.grid('on')
    #plt.axhline(linewidth=2, color='black')
    plt.title('Sea Level forecast and observations at '+tg,fontsize=18) #and diff wrt obs in '+tg,fontsize=18)
    plt.ylim(0,200)
    if time_p == 'allp' :
       plt.xlim([datetime(2019,11,9,0,0,0),datetime(2019,11,12,23,30,0)])
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
       #ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%H"))
       ax.margins(x=0)
    elif time_p == 'osr5' :
       plt.xlim([datetime(2019,11,10,0,0,0),datetime(2019,11,12,23,30,0)])
       plt.xlabel ('November 2019',fontsize=16)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       ax.margins(x=0)
       plt.ylim(0,200)




    plt.tight_layout()
    plt.savefig(fig_name,format='png') #,dpi=1200)
    plt.clf()

  

