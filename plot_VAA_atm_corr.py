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
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
mpl.use('Agg')
#####################################

# -- Workdir path -- 
workdir   = '/work/oda/ag15419/tmp/Venezia_Acqua_Alta/VAA_prec_extr/'
print ('workdir',workdir)

file_ssh  = 'ISMAR_TG_mod_EAS6_FC3_w10.nc'
file_prec = 'ISMAR_TG_mod_prec_EAS56_FC3_w10.nc'
file_pres = 'ISMAR_TG_mod_pres_EAS56_FC3_w10.nc' 
file_wind = 'ISMAR_TG_mod_wind_EAS56_FC3_w10.nc'

ssh = ncdf.Dataset(workdir+file_ssh,mode='r')
prec = ncdf.Dataset(workdir+file_prec,mode='r')
pres = ncdf.Dataset(workdir+file_pres,mode='r')
wind = ncdf.Dataset(workdir+file_wind,mode='r')
 
ssh_field  = ssh.variables['sossheig'][:]
prec_field = prec.variables['precip'][:]
pres_field = pres.variables['MSL'][:]
wind_field = wind.variables['W10'][:]

time_ssh   = ssh.variables['time_counter'][:]
time_ssh_units = ssh.variables['time_counter'].getncattr('units')

time_atm   = prec.variables['time'][:]
time_atm_units = prec.variables['time'].getncattr('units')

ssh.close()
prec.close()
pres.close()
wind.close()

alltimes_ssh=[]
for alltime_idx in range (0,len(time_ssh)):
    alltimes_ssh.append(datetime(ncdf.num2date(time_ssh[alltime_idx],time_ssh_units).year,ncdf.num2date(time_ssh[alltime_idx],time_ssh_units).month,ncdf.num2date(time_ssh[alltime_idx],time_ssh_units).day,ncdf.num2date(time_ssh[alltime_idx],time_ssh_units).hour,ncdf.num2date(time_ssh[alltime_idx],time_ssh_units).minute,ncdf.num2date(time_ssh[alltime_idx],time_ssh_units).second))

alltimes_atm=[]
for alltime_idx in range (0,len(time_ssh)):
    alltimes_atm.append(datetime(ncdf.num2date(time_atm[alltime_idx],time_atm_units).year,ncdf.num2date(time_atm[alltime_idx],time_atm_units).month,ncdf.num2date(time_atm[alltime_idx],time_atm_units).day,ncdf.num2date(time_atm[alltime_idx],time_atm_units).hour,ncdf.num2date(time_atm[alltime_idx],time_atm_units).minute,ncdf.num2date(time_atm[alltime_idx],time_atm_units).second))

def func4fit_pr(t, a, b, c, d):

    for ti in range (0,len(t)):
       p = pres_field[ti] / 1000
       w = wind_field[ti] / 10
       r = prec_field[ti] * 100
    return np.squeeze(a * p + b * w + c * r + d)

def func4fit_nopr(t, e, f, g):

    for ti in range (0,len(t)):
       p = pres_field[ti] / 1000
       w = wind_field[ti] /10
    return np.squeeze(e * p + f * w + g)

popt_pr, pcov_pr     = curve_fit(func4fit_pr,np.arange(0,len(time_ssh)),np.squeeze(ssh_field))
a      = popt_pr[0]/1000
b      = popt_pr[1]/10
c      = popt_pr[2]*100
d      = popt_pr[3]
fit_pr = np.squeeze(a*pres_field[0:len(time_ssh)]+b*wind_field[0:len(time_ssh)]+c*prec_field[0:len(time_ssh)]+d)
print ('fit prec',a,b,d,d)

popt_nopr, pcov_nopr = curve_fit(func4fit_nopr,np.arange(0,len(time_ssh)),np.squeeze(ssh_field))
e        = popt_nopr[0]/1000
f        = popt_nopr[1]/10
g        = popt_nopr[2]
fit_nopr = np.squeeze(e*pres_field[0:len(time_ssh)]+f*wind_field[0:len(time_ssh)]+g)
print ('fit no prec',e,f,g)

# Plot
plt.figure(figsize=(12,6))
plt.rc('font', size=16)
plt.title ('SSH Corr with atm fields')
plt.plot(alltimes_ssh,np.squeeze(ssh_field)-np.mean(np.squeeze(ssh_field)),label='SSH [m]')
#plt.plot(alltimes_ssh,np.squeeze((pres_field[0:len(time_ssh)]-np.mean(np.squeeze(pres_field[0:len(time_ssh)])))/1000),label='MSL /10000 [Pa] ')
#plt.plot(alltimes_ssh,np.squeeze((wind_field[0:len(time_ssh)]-np.mean(np.squeeze(wind_field[0:len(time_ssh)])))/10),label='W10 /10 [m/s]')
#plt.plot(alltimes_ssh,np.squeeze(100*(prec_field[0:len(time_ssh)]-np.mean(np.squeeze(prec_field[0:len(time_ssh)])))),label='precip *100 [m/h]')
plt.plot(alltimes_ssh,fit_pr-np.mean(fit_pr),'-',color='red',label='Fit with prec')
plt.plot(alltimes_ssh,fit_nopr-np.mean(fit_nopr),'--',color='red',label='Fit without prec')
plt.grid ()
#plt.ylabel ('m')
plt.xlabel ('Date')
plt.legend()
plt.tight_layout()
plt.savefig(workdir+'SSH_corr.png',format='png')
plt.clf()

c_pres = np.corrcoef(np.squeeze(ssh_field), np.squeeze(pres_field[0:len(time_ssh)]))
c_wind = np.corrcoef(np.squeeze(ssh_field), np.squeeze(wind_field[0:len(time_ssh)]))
c_prec = np.corrcoef(np.squeeze(ssh_field), np.squeeze(prec_field[0:len(time_ssh)]))


print ('Corr SSH-Pres',c_pres)
print ('Corr SSH-Wind',c_wind)
print ('Corr SSH-Prec',c_prec)
