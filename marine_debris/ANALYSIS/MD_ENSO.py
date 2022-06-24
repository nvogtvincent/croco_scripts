#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carry out EOF analysis on MD data
@author: noam
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmasher as cmr
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from matplotlib.gridspec import GridSpec
from sys import argv
from scipy import signal

# PARAMETERS
param = {'degrade': 12,                            # Degradation factor
         'lon_range': [20, 130],                   # Longitude range for output
         'lat_range': [-40, 30],                   # Latitude range for output

         # Analysis parameters
         'us_d': int(argv[1]),    # Sinking timescale (days)
         'ub_d': int(argv[2]),    # Beaching timescale (days)

         # Physics
         'physics': argv[3],

         # Source/sink time
         'time': argv[4],

         # Sink sđites
         'sites': np.array([1,2,3,4]),

         # Which mode
         'mode': argv[5], # IOD/NINO34

         # Filtering parameters
         'delay': int(argv[6]), # Delay (months) for climate index, i.e. leads observations
         'crit_freq': 12/16,

         # Set significance threshold (for log transform)
         'sig_thresh': 1e-9,}

try:
    param['name'] = argv[7] + ' '
except:
    param['name'] = ''

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'ref': os.path.dirname(os.path.realpath(__file__)) + '/../REFERENCE/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
array_str = np.array2string(param['sites'], separator='-').translate({ord(i): None for i in '[]'})
fh = {'flux': dirs['script'] + '/marine_' + param['time'] + '_flux_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'DMI': dirs['ref'] + 'dmi.had.long.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/
      'NINO4': dirs['ref'] + 'nino4.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino4/
      'NINO34': dirs['ref'] + 'nino34.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino34/
      'NINO3': dirs['ref'] + 'nino3.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino3/
      'critical_values': dirs['ref'] + 'critical_values.txt'
      }

time_var = 'sink_time' if param['time'] == 'sink' else 'source_time'
dof_mult = {'NINO4': 8, 'NINO34': 7, 'NINO3': 7, 'DMI': 4}

##############################################################################
# LOAD DATA                                                                  #
##############################################################################

# Only use 1995-2012 for sink (2-year spinup, NOT valid for us > 1 year)
if param['time'] == 'sink':
    fmatrix = xr.open_dataarray(fh['flux'])[:, :, 24:240]
else:
    fmatrix = xr.open_dataarray(fh['flux'])[:]

# Degrade resolution
fmatrix = fmatrix.coarsen(dim={'longitude': param['degrade'], 'latitude': param['degrade'], time_var: 1}).sum()
fmatrix = fmatrix.transpose(time_var, 'longitude', 'latitude')

lons, lats = np.meshgrid(fmatrix.coords['longitude'], fmatrix.coords['latitude'])

# Log transform data:
# We release 36*4*(12^2) particles per degree grid cell (except for edge cases)
# so let's set everything less than 0.1% of this to 'negligible'.
sig_thresh = 36*4*param['degrade']*param['degrade']*param['sig_thresh']
fmatrix = fmatrix.where(fmatrix > sig_thresh)
fmatrix = fmatrix.fillna(sig_thresh)
fmatrix = np.log10(fmatrix)

# Pass fmatrix through a low-pass filter
input_freq = 12 # (1/yr)
seas_freq = 1   # (1/yr)

lp = signal.butter(3, param['crit_freq'], fs=12, btype='lowpass', analog=False, output='sos')
fmatrix_lp = fmatrix.copy()
fmatrix_lp.values = signal.sosfiltfilt(lp, fmatrix.values, axis=0)

# Load the chosen climate index
clim = pd.read_table(fh[param['mode']], skiprows=1, delim_whitespace=True,
                     header=None, index_col=0, skipfooter=8, engine='python')

if param['time'] == 'sink':
    clim = clim[clim.index.isin(np.arange(1995-2,2013+2))].values.flatten()[24-param['delay']:-24-param['delay']]
    clim = xr.DataArray(clim, coords={'sink_time': fmatrix.coords['sink_time']})
else:
    clim = clim[clim.index.isin(np.arange(1993-2,2013+2))].values.flatten()[24-param['delay']:-24-param['delay']]
    clim = xr.DataArray(clim, coords={'source_time': fmatrix.coords['source_time']})

##############################################################################
# CALCULATE CORRELATION                                                      #
##############################################################################

if param['time'] == 'sink':
    corr = xr.corr(fmatrix_lp, clim, dim='sink_time')
else:
    corr = xr.corr(fmatrix_lp, clim, dim='source_time')

# Assess significance
n_obs = len(fmatrix_lp.coords['sink_time']) if param['time'] == 'sink' else len(fmatrix_lp.coords['source_time'])
critical_table = pd.read_table(fh['critical_values'], delim_whitespace=True,
                               index_col=0, skipfooter=1, engine='python').loc[:, '0.05']

r_crit = critical_table[int(n_obs/(2*dof_mult[param['mode']]))]

##############################################################################
# PLOT                                                                       #
##############################################################################

f = plt.figure(figsize=(20, 11.5), constrained_layout=True)
gs = GridSpec(1, 2, figure=f, width_ratios=[0.99, 0.01])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())) # Main figure (eof)
ax.append(f.add_subplot(gs[0, 1])) # Colorbar for main figure

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='white',
                                        facecolor='black',
                                        zorder=1)
ax[0].set_aspect(1)

# Plot EOF
corr_plot = ax[0].contourf(fmatrix_lp.coords['longitude'], fmatrix_lp.coords['latitude'], corr.T,
                           cmap=cmr.fusion_r, transform=ccrs.PlateCarree(), extend='neither',
                           levels=np.linspace(-1, 1, 21), rasterized=True)
sig_plot = ax[0].contourf(fmatrix_lp.coords['longitude'], fmatrix_lp.coords['latitude'], np.abs(corr.T),
                          levels=np.array([r_crit, 1]), hatches=['.'], colors='none')

ax[0].add_feature(land_10m)
gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='white', linestyle='--', zorder=11)
gl.xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl.xlabels_top = False
gl.ylabels_right = False
gl.ylabel_style = {'size': 24}
gl.xlabel_style = {'size': 24}

title = param['mode'] + ' correlation'
ax[0].text(21, -39, title, fontsize=32, color='k', fontweight='bold')
ax[0].set_xlim([20, 130])
ax[0].set_ylim([-40, 30])

if param['time'] == 'sink':
    ax[0].set_title(param['name'] + ' (Sink time)', fontsize=32, color='k', fontweight='bold')
else:
    ax[0].set_title(param['name'] + ' (Source time)', fontsize=32, color='k', fontweight='bold')

cb0 = plt.colorbar(corr_plot, cax=ax[1], fraction=0.1)
ax[1].tick_params(axis='y', labelsize=24)
cb0.set_label('Correlation', size=28)
cb0.set_ticks(np.linspace(-1, 1, 5))

# Save
plt.savefig(dirs['fig'] + param['mode'] + '_LP_' + param['time'] + '_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf', bbox_inches='tight', dpi=300)