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
from datetime import datetime,timedelta
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
workdir = '/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_wind_new/'

# -- Period --
start_date = 20191112 #12 #09
end_date   = 20191115 #13 #17

# length of the time interval to be plotted: allp, zoom or super-zoom
time_p = 'allp'

# To interpolate the obs from hh:00 to hh:30 and plot mod-obs diffs
obs_interp_flag = 0

# ---  Input archive ---
input_dir          = '/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_wind_new/'
#
input_tg   = ['ISMAR_TG']
input_dat  = ['mod_wind','obs_wind'] # Do not change the order because the obs are used as reference for offset and differences!
input_type = ['FC1','FC2','FC3','AN']
input_res  = ['08','08sub','10'] # Do not change the order 
input_sys  = ['EAS4','EAS5','EAS6','EAS56'] # The last must be 'EAS56'

input_var     = 'W10' 
udm           = 'm/s'
input_mod_timevar = 'time'

# Color
colors = pl.cm.jet_r(np.linspace(0,1,13)) #13
colors = pl.cm.Purples(np.linspace(0.1,0.9,13))

#############################
# Defn running mean for obs
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# Loop on tide-gauges
for tg_idx,tg in enumerate(input_tg):

    # Output file
    fig_name = workdir+'/'+tg+'_'+time_p+'_wind.png' #'zoom.png'

    # Loop on datasets
    for dat_idx,dat in enumerate(input_dat):

        # OBS
        if dat == 'obs_wind':
           # input files name
           file_to_open = input_dir+'/'+tg+'_'+dat+'.csv' #+'.nc'
           print ('Open files: ',file_to_open)
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
              print ('Obs time',alltimes_obs)

              # Read obs time series
              #var_obs  = fh.variables[input_var][:]
              var_obs = fh[3][:]
              var_obs = np.array(var_obs) #/ 100.0
              #print ('Pre',len(var_obs))
              #var_obs = running_mean(var_obs,12)
              #var_obs = var_obs[::12]
              #print ('Post',len(var_obs))
              # Interpolate from :00 to :30
              if obs_interp_flag == 1:
                 where_to_interp = np.linspace(0.5,float(len(var_obs))+0.5,216)
                 interp_obs = np.interp(linspace(0.5,len(var_obs)+0.5,216),range(0,len(var_obs)),var_obs)
                 var_obs = interp_obs

              # Close infile 
              #fh.close()
           else:
              print ('NOT Found!')  
        # MOD
        elif dat == 'mod_wind':
            # Loop on system versions
            for easys_idx,easys in enumerate(input_sys):
                # Loop on ts type
                for atype_idx,atype in enumerate(input_type):
                    # Loop on atm forcing spatial resolutions
                    for res_idx,res in enumerate(input_res):
                        print ('Working on',tg,dat,easys,atype,res)
                        # input file name
                        #infile = globals()[tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res+'.nc']
                        file_to_open = input_dir+'/'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res+'.nc'
                        print ('Open file: ',file_to_open)
                        # check the existence of the file and open it
                        print ('Try: ',file_to_open+'_ok.nc')
                        if glob.glob(file_to_open+'_ok.nc'):
                           fh = ncdf.Dataset(file_to_open+'_ok.nc',mode='r')
                           # Read time axes and compute time-var
                           time_mod   = fh.variables[input_mod_timevar][:]
                           time_mod_units = fh.variables[input_mod_timevar].getncattr('units')
                           globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=[]
                           for alltime_idx in range (0,len(time_mod)):
                               globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res].append(datetime(ncdf.num2date(time_mod[alltime_idx],time_mod_units).year,ncdf.num2date(time_mod[alltime_idx],time_mod_units).month,ncdf.num2date(time_mod[alltime_idx],time_mod_units).day,ncdf.num2date(time_mod[alltime_idx],time_mod_units).hour,ncdf.num2date(time_mod[alltime_idx],time_mod_units).minute,ncdf.num2date(time_mod[alltime_idx],time_mod_units).second))
                           globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res] = np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                           print ('Time',np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][0]))
                           # Save the time for obs
                           alltimes_mod_tmp = globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]

                           # Read mod time series
                           fh = ncdf.Dataset(file_to_open+'_ok.nc',mode='r')
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = fh.variables[input_var][:] 

                           # Squeeze the array and print the first element
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res] = np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                           print ('Var',np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[0])
 

#                           # Compute the differences wrt obs
#                           globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])-var_obs

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
                           globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res] = np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                           print ('Time',np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][0]))

                           # Read mod time series
                           fh = ncdf.Dataset(file_to_open,mode='r')
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = fh.variables[input_var][:]

                           # Squeeze the array and print the first element
                           globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res] = np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])
                           print ('Var',np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[0])

                           # Close infile
                           fh.close()

#                           # Compute the differences wrt obs
#                           globals()['diff_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]=np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])-var_obs
                        else:
                           print ('NOT Found!')


######## PLOT TS #########
# Loop on tide-gauges
for tg_idx,tg in enumerate(input_tg):
    # Initialize the plot
    fig = plt.figure(0,figsize=(20,11))
    print ('Plot: ',fig_name)
    plt.rc('font', size=16)
#    if obs_interp_flag == 1:
#       fig.add_subplot(111)
#       gs = fig.add_gridspec(2, 3)
#       # Abs values
#       ax = plt.subplot(gs[0, :-1]) #(2,2,1)
#    else:
    fig.add_subplot(111)
    gs = fig.add_gridspec(1, 3)
    ax = plt.subplot(gs[0, :-1])
    # OBS
    alltimes_obs_frommod = np.arange(alltimes_mod_tmp[0],alltimes_mod_tmp[-1], timedelta(hours=1)).astype(datetime)
    max_val=np.max(var_obs[:-3])
    ax.plot(alltimes_obs_frommod,var_obs[:-3],'-',color='red',label='OBS '+' (max: '+str(round(max_val,1))+' m/s)',linewidth=3)

    # Line index
    idx_line_plot = 0
    # Loop on datasets
    for dat_idx,dat in enumerate(input_dat):
        # MOD
        if dat == 'mod_wind':
            # Loop on system versions
            for easys_idx,easys in enumerate(input_sys):
                # Loop on ts type
                for atype_idx,atype in enumerate(input_type):
                    # Loop on atm forcing spatial resolutions
                    for res_idx,res in enumerate(input_res):

                        # Plot the lines corresponding to existing datasets!
                        file_to_open = input_dir+'/'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res+'.nc'
                        if glob.glob(file_to_open):
                           print ('PLOTTING: ',file_to_open)
                           print ('T',np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[0])
                           print ('V',np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[0])
                           print ('Col/Lab',idx_line_plot,' ',easys,'_',atype,'_w',res)

                           if atype != 'AN' and res == '10' :
                              print (len(np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])))
                              max_val=np.max(np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][:-23])) # 72
                           elif atype != 'AN' and res != '10' :
                              max_val=np.max(np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][:-24]))
                           else:
                              max_val=np.max(np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res][:-12]))

                           if easys != 'EAS4':
                              labels='ECMWF '+atype+'_w'+res+' (max: '+str(round(max_val,1))+' m/s)'   #+' ('+atype+' '+easys+' atm forc)'
                           else:
                              labels='ECMWF '+atype+'_w'+res+'_12'+' (max: '+str(round(max_val,1))+' m/s)' #+' ('+atype+' '+easys+' atm forc)'
                           print ('labels',labels)

                           if atype == 'AN':
                              linetype='-'
                           elif atype != 'AN' and res != '10':
                              linetype='--'
                           else:
                              linetype=':'

                           ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),linetype,color=colors[idx_line_plot],label=labels,linewidth=3)

                           #ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'-',color=colors[idx_line_plot],label='ECMWF '+atype+'_w'+res+' ('+atype+' '+easys+' atm forc)',linewidth=1.5) 
                           print ('Plotting',tg,dat,easys,atype,res)

                           # Update line in plot index
                           idx_line_plot = idx_line_plot + 1


    # Finalize the plot
    ylabel("W10M Speed [m/s]",fontsize=18)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,  shadow=True, fancybox=True, fontsize=12)
    ##leg = plt.legend(loc='lower right', ncol=2,  shadow=True, fancybox=True, fontsize=12)
    leg = plt.legend(loc='upper center',ncol=1,  shadow=True, fancybox=True, fontsize=12)
    ##leg.get_frame().set_alpha(0.3)
    ax.grid('on')
    #plt.axhline(linewidth=2, color='black')
    plt.title('W10M Speed time-series at '+tg,fontsize=18)
    if time_p == 'allp' :
       plt.xlim([datetime(2019,11,12,0,0,0),datetime(2019,11,15,23,30,0)])
       plt.xlabel ('November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       ax.margins(x=0)
       plt.ylim(0,30)
    elif time_p == 'zoom' :
       plt.xlim([datetime(2019,11,12,0,0,0),datetime(2019,11,13,23,30,0)])
       plt.xlabel ('November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.DayLocator())
       ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d"))
       #ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%H"))
       ax.margins(x=0)
    elif time_p == 'superzoom' :
       plt.ylim(2.5,20)
       plt.xlim([datetime(2019,11,12,16,30,0),datetime(2019,11,12,23,30,0)])
       plt.xlabel ('12 November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.HourLocator())
       #ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%H"))
       ax.margins(x=0)


    plt.tight_layout()
    plt.savefig(fig_name,format='png',dpi=1200)
    plt.clf()

  

