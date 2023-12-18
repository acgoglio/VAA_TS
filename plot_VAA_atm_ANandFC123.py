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
workdir = '/work/oda/med_dev/VAA_w08_w10_ANFC_plot_atm/'

# -- Period --
start_date = 20191109 #12 #09
end_date   = 20191113 #13 #17

# length of the time interval to be plotted: allp, zoom or super-zoom
time_p = 'allp'

# To interpolate the obs from hh:00 to hh:30 and plot mod-obs diffs
obs_interp_flag = 0

# ---  Input archive ---
input_dir          = '/work/oda/med_dev/VAA_w08_w10_ANFC_atm/'
#
input_tg   = ['ISMAR_TG']
input_dat  = ['mod_atm','obs_atm'] # Do not change the order because the obs are used as reference for offset and differences!
input_type = ['FC1','FC2','FC3','AN']
input_res  = ['08'] #,'08sub','10'] # Do not change the order 
input_sys  = ['EAS4','EAS5','EAS6','EAS56'] # The last must be 'EAS56'

input_var     = 'MSL' 
udm           = 'hPa'
input_mod_timevar = 'time'

# Color
colors = pl.cm.jet_r(np.linspace(0,1,8)) #13

#############################
# Loop on tide-gauges
for tg_idx,tg in enumerate(input_tg):

    # Output file
    fig_name = workdir+'/'+tg+'_'+time_p+'_atm.png' #'zoom.png'

    # Loop on datasets
    for dat_idx,dat in enumerate(input_dat):

        # OBS
        if dat == 'obs_atm':
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
              var_obs = fh[1][:] #*100.0
              var_obs = np.array(var_obs)
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
        elif dat == 'mod_atm':
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

                           # For AN convert Pa -> hPa
                           if atype == 'AN':
                              globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  / 100

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

                           # For AN convert Pa -> hPa
                           if atype == 'AN':
                              globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  = globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]  / 100
                          
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
    fig = plt.figure(0,figsize=(16,8))
    print ('Plot: ',fig_name)
#    if obs_interp_flag == 1:
#       fig.add_subplot(111)
#       gs = fig.add_gridspec(2, 3)
#       # Abs values
#       ax = plt.subplot(gs[0, :-1]) #(2,2,1)
#    else:
    fig.add_subplot(111)
    gs = fig.add_gridspec(1, 3)
    ax = plt.subplot(gs[0, :-1])
    # Line index
    idx_line_plot = 0
    # Loop on datasets
    for dat_idx,dat in enumerate(input_dat):
        # MOD
        if dat == 'mod_atm':
            # Loop on system versions
            for easys_idx,easys in enumerate(input_sys):
                # Loop on ts type
                for atype_idx,atype in enumerate(input_type):
                    # Loop on atm forcing spatial resolutions
                    for res_idx,res in enumerate(input_res):

                        # Plot the lines corresponding to existing datasets!
                        file_to_open = input_dir+'/'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res+'.nc'
                        if glob.glob(file_to_open):
                           print ('Prova')
                           print ('T',np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[0])
                           print ('V',np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res])[0])
                           print ('Col/Lab',idx_line_plot,' ',easys,'_',atype,'_w',res)

                           ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'-',color=colors[idx_line_plot],label='ECMWF '+atype+'_w'+res+' ('+atype+' '+easys+' atm forc)',linewidth=1.5) 
                           print ('Plotting',tg,dat,easys,atype,res)

#                           # Plot the mod lines in the plot (line type based on time res)
#                           if res == '10' and (atype == 'FC1' or atype == 'FC2' or atype == 'FC3') :
#                              try:
#                                 ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),':',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
#                                 print ('Plotting',tg,dat,easys,atype,res)
#                              except:
#                                 print ('No')
#                           elif res == '08' and (atype == 'FC1' or atype == 'FC2' or atype == 'FC3') :
#                           #try:
#                           #   print ('Prova',tg,dat,easys,atype,res)
#                           #   print ('Prova',np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]))
#                              try:
#                                 ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),'--',color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
#                                 print ('Plotting',tg,dat,easys,atype,res)
#                              except:
#                                 print ('No')
#                           else:
#                              try:
#                                 ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),np.squeeze(globals()['var_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),color=colors[idx_line_plot],label=easys+'_'+atype+'_w'+res,linewidth=1.5)
#                                 print ('Plotting',tg,dat,easys,atype,res)
#                              except:
#                                 print ('No')

                           # Update line in plot index
                           idx_line_plot = idx_line_plot + 1


    # OBS
    alltimes_obs_frommod = np.arange(alltimes_mod_tmp[0],alltimes_mod_tmp[-1], timedelta(hours=1)).astype(datetime)
    ax.plot(alltimes_obs_frommod,var_obs[:-3],'o-',color='black',label='OBS',linewidth=1.5)
    #ax.plot(np.squeeze(globals()['alltimes_mod_'+tg+'_'+dat+'_'+easys+'_'+atype+'_w'+res]),var_obs,'o-',color='black',label='OBS',linewidth=1.5)

    # Finalize the plot
    ylabel("MSL [hPa]",fontsize=18)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,  shadow=True, fancybox=True, fontsize=12)
    ##leg = plt.legend(loc='lower right', ncol=2,  shadow=True, fancybox=True, fontsize=12)
    ##leg.get_frame().set_alpha(0.3)
    ax.grid('on')
    #plt.axhline(linewidth=2, color='black')
    plt.title('ECMWF MSL time-series '+tg,fontsize=18)
    if time_p == 'allp' :
       plt.xlim([datetime(2019,11,9,0,0,0),datetime(2019,11,15,23,30,0)])
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
       plt.ylim(985,1000)
       plt.xlim([datetime(2019,11,12,16,30,0),datetime(2019,11,12,23,30,0)])
       plt.xlabel ('12 November 2019',fontsize=18)
       ax.xaxis.set_major_locator(mdates.HourLocator())
       #ax.xaxis.set_minor_locator(mdates.HourLocator((6,12,18)))
       ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%H"))
       ax.margins(x=0)


    plt.tight_layout()
    plt.savefig(fig_name,format='png',dpi=1200)
    plt.clf()

  

