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
from scipy.fft import rfft, rfftfreq

# PARAMETERS
param = {'degrade': 12,                            # Degradation factor
         'lon_range': [20, 130],                   # Longitude range for output
         'lat_range': [-40, 30],                   # Latitude range for output

         # Analysis parameters
         'us_d': 1800,    # Sinking timescale (days)
         'ub_d': 20,    # Beaching timescale (days)

         # Physics
         'physics': '0000NS',

         # Source/sink time
         'time': 'sink',

         # Sink sđites
         'sites': np.array(np.arange(1, 19)),
         'name': 'Seychelles',

         # Log options
         'log': True,
         'sig_thresh': 1e-9,}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'ref': os.path.dirname(os.path.realpath(__file__)) + '/../REFERENCE/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
array_str = np.array2string(param['sites'], separator='-').translate({ord(i): None for i in '[]'})
fh = {'flux': dirs['script'] + '/marine_' + param['time'] + '_flux_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'critical_values': dirs['ref'] + 'critical_values.txt'
      }

time_var = 'sink_time' if param['time'] == 'sink' else 'source_time'

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
# if param['log']:
#     sig_thresh = 36*4*param['degrade']*param['degrade']*param['sig_thresh']
#     fmatrix = fmatrix.where(fmatrix > sig_thresh)
#     fmatrix = fmatrix.fillna(sig_thresh)
#     fmatrix = np.log10(fmatrix)

##############################################################################
# CARRY OUT FOURIER TRANSFORM                                                #
##############################################################################

yf = rfft(fmatrix.values, axis=0)
xf = rfftfreq(len(fmatrix.coords[time_var]), 1/12)

seas_freq_idx = np.where(xf == 1)
phase = -np.angle(yf[seas_freq_idx, :, :])[0, 0, :, :]*6/np.pi

# Convert phase to peak month (0 -> Jan, 1 -> Feb, -1 -> Dec...)
phase[phase < 0] = phase[phase < 0] + 12

# Also calculate based on monthly climatology
monclim = np.zeros((12, np.shape(fmatrix)[1], np.shape(fmatrix)[2]))

for mo in range(12):
    monclim[mo, :, :] = np.mean(fmatrix.values[mo::12, :, :], axis=0)

min_mon = np.argmin(monclim, axis=0)+0.5
max_mon = np.argmax(monclim, axis=0)+0.5

##############################################################################
# PLOT                                                                       #
##############################################################################
f = plt.figure(figsize=(20, 23), constrained_layout=True)
gs = GridSpec(2, 2, figure=f, width_ratios=[0.99, 0.02])
ax = []
gl = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())) # Peak
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree())) # Min
ax.append(f.add_subplot(gs[:, 1])) # Colorbar

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='white',
                                        facecolor='black',
                                        zorder=1)
ax[0].set_aspect(1)

# Plot peak month
max_plot = ax[0].pcolormesh(np.append(fmatrix.coords['longitude'], fmatrix.coords['longitude'][-1]+1)-0.5,
                            np.append(fmatrix.coords['latitude'], fmatrix.coords['latitude'][-1]+1)-0.5, max_mon.T,
                            vmin=0, vmax=12, cmap=cmr.infinity, transform=ccrs.PlateCarree(), rasterized=True)

ax[0].add_feature(land_10m)
gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='white', linestyle='--', zorder=11))
gl[0].xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[0].xlabels_top = False
gl[0].ylabels_right = False
gl[0].ylabel_style = {'size': 24}
gl[0].xlabel_style = {'size': 24}

ax[0].set_xlim([20, 130])
ax[0].set_ylim([-40, 30])
ax[0].set_title('Predicted maximum dFAD beaching month at ' + param['name'], fontsize=32, color='k', fontweight='bold')

# Plot minimum month
min_plot = ax[1].pcolormesh(np.append(fmatrix.coords['longitude'], fmatrix.coords['longitude'][-1]+1)-0.5,
                            np.append(fmatrix.coords['latitude'], fmatrix.coords['latitude'][-1]+1)-0.5, min_mon.T,
                            cmap=cmr.infinity, transform=ccrs.PlateCarree(), rasterized=True)

ax[1].add_feature(land_10m)
gl.append(ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='white', linestyle='--', zorder=11))
gl[1].xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
gl[1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[1].xlabels_top = False
gl[1].ylabels_right = False
gl[1].ylabel_style = {'size': 24}
gl[1].xlabel_style = {'size': 24}

ax[1].set_xlim([20, 130])
ax[1].set_ylim([-40, 30])
ax[1].set_title('Predicted minimum dFAD beaching month at ' + param['name'], fontsize=32, color='k', fontweight='bold')

cb0 = plt.colorbar(max_plot, cax=ax[2], fraction=0.1)
ax[2].tick_params(axis='y', labelsize=24)
cb0.set_ticks(np.linspace(0.5, 11.5, 12))
cb0.set_ticklabels(['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December'])
# Save
plt.savefig(dirs['fig'] + 'dFAD_beaching_times_' + param['name'] + '_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf', bbox_inches='tight', dpi=300)
# f = plt.figure(figsize=(20, 11.5), constrained_layout=True)
# gs = GridSpec(1, 2, figure=f, width_ratios=[0.99, 0.02])
# ax = []
# ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())) # Main figure (eof)
# ax.append(f.add_subplot(gs[0, 1])) # Colorbar for main figure

# land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
#                                         edgecolor='white',
#                                         facecolor='black',
#                                         zorder=1)
# ax[0].set_aspect(1)

# # Plot EOF
# corr_plot = ax[0].pcolormesh(np.append(fmatrix.coords['longitude'], fmatrix.coords['longitude'][-1]+1)-0.5,
#                              np.append(fmatrix.coords['latitude'], fmatrix.coords['latitude'][-1]+1)-0.5, peak_mon.T,
#                              cmap=cmr.infinity, transform=ccrs.PlateCarree(), rasterized=True)
# # sig_plot = ax[0].contourf(fmatrix_lp.coords['longitude'], fmatrix_lp.coords['latitude'], np.abs(corr.T),
# #                           levels=np.array([r_crit, 1]), hatches=['.'], colors='none')

# ax[0].add_feature(land_10m)
# gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                       linewidth=0.5, color='white', linestyle='--', zorder=11)
# gl.xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
# gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.ylabel_style = {'size': 24}
# gl.xlabel_style = {'size': 24}

# ax[0].set_xlim([20, 130])
# ax[0].set_ylim([-40, 30])
# ax[0].set_title('Predicted peak dFAD beaching month at Seychelles', fontsize=32, color='k', fontweight='bold')

# cb0 = plt.colorbar(corr_plot, cax=ax[1], fraction=0.1)
# ax[1].tick_params(axis='y', labelsize=24)
# cb0.set_label('Peak month', size=28)
# cb0.set_ticks(np.linspace(0.5, 11.5, 12))
# cb0.set_ticklabels(['January', 'February', 'March', 'April', 'May', 'June',
#                     'July', 'August', 'September', 'October', 'November', 'December'])
# # Save
# plt.savefig(dirs['fig'] + 'MacMillan_corr_' + param['time'] + '_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf', bbox_inches='tight', dpi=300)