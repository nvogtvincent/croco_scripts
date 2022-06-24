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
         'us_d': 1800,    # Sinking timescale (days)
         'ub_d': 30,    # Beaching timescale (days)

         # Physics
         'physics': '0050',

         # Source/sink time
         'time': 'sink',

         # Sink sđites
         'sites': np.array([13,14,15,16,17,18]),

         # Seasonal cycle
         'seas': np.array([43,44,54,58,38,20,16,24,33,40,31,38]),

         # Log options
         'log': True,
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
if param['log']:
    sig_thresh = 36*4*param['degrade']*param['degrade']*param['sig_thresh']
    fmatrix = fmatrix.where(fmatrix > sig_thresh)
    fmatrix = fmatrix.fillna(sig_thresh)
    fmatrix = np.log10(fmatrix)

# Generate a monthly climatology
n_year = int(len(fmatrix.coords[time_var])/12)
seas = xr.DataArray(np.tile(param['seas'], n_year)/np.max(param['seas']), coords={time_var: fmatrix.coords[time_var]})

##############################################################################
# CALCULATE CORRELATION                                                      #
##############################################################################

corr = xr.corr(fmatrix, seas, dim=time_var)

# Assess significance
# n_obs = len(fmatrix_lp.coords['sink_time']) if param['time'] == 'sink' else len(fmatrix_lp.coords['source_time'])
# critical_table = pd.read_table(fh['critical_values'], delim_whitespace=True,
#                                index_col=0, skipfooter=1, engine='python').loc[:, '0.05']

# r_crit = critical_table[int(n_obs/(2*dof_mult[param['mode']]))]

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
corr_plot = ax[0].contourf(fmatrix.coords['longitude'], fmatrix.coords['latitude'], corr.T,
                           cmap=cmr.fusion_r, transform=ccrs.PlateCarree(), extend='neither',
                           levels=np.linspace(-1, 1, 21), rasterized=True)
# sig_plot = ax[0].contourf(fmatrix_lp.coords['longitude'], fmatrix_lp.coords['latitude'], np.abs(corr.T),
#                           levels=np.array([r_crit, 1]), hatches=['.'], colors='none')

ax[0].add_feature(land_10m)
gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='white', linestyle='--', zorder=11)
gl.xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl.xlabels_top = False
gl.ylabels_right = False
gl.ylabel_style = {'size': 24}
gl.xlabel_style = {'size': 24}

title = 'dFAD correlation'
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
plt.savefig(dirs['fig'] + 'MacMillan_corr_' + param['time'] + '_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf', bbox_inches='tight', dpi=300)