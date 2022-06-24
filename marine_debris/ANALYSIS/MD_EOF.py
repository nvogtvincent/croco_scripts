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
from matplotlib.gridspec import GridSpec
from eofs.xarray import Eof
from sys import argv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

         # EOF mode
         'eof_mode': 0,

         # Sink sites
         'sites': np.array([1,2,3,4]),

         # Set significance threshold (for log transform)
         'sig_thresh': 1e-9,

         # Export
         'export': True}

try:
    param['name'] = argv[5] + ' '
except:
    param['name'] = ''

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'fisheries': os.path.dirname(os.path.realpath(__file__)) + '/../FISHERIES/DATA/PROC/',
        'shipping': os.path.dirname(os.path.realpath(__file__)) + '/../SHIPPING/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
array_str = np.array2string(param['sites'], separator='-').translate({ord(i): None for i in '[]'})
fh = {'flux': dirs['script'] + '/marine_' + param['time'] + '_flux_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'drift_time': dirs['script'] + '/marine_' + param['time'] + '_drift_time_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'debris': dirs['fisheries'] + 'GFW_gridded.nc',
      'shipping': dirs['shipping'] + 'shipping_2008.tif'}

time_var = 'sink_time' if param['time'] == 'sink' else 'source_time'

##############################################################################
# LOAD DATA                                                                  #
##############################################################################

# Only use 1995-2012 (2-year spinup, NOT valid for us > 1 year)
if param['time'] == 'sink':
    fmatrix = xr.open_dataarray(fh['flux'])[:, :, 24:240]
else:
    fmatrix = xr.open_dataarray(fh['flux'])[:, :, :240]

# Degrade resolution to 1deg
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

##############################################################################
# CALCULATE EOFs                                                             #
##############################################################################

# Remove the mean to calculate anomalies
fmatrix_anom = fmatrix - fmatrix.mean(dim=time_var)

# Create an EOF solver with sqrt(cos) weights following
# https://ajdawson.github.io/eofs/latest/examples/nao_xarray.html
coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(fmatrix_anom, weights=wgts)

# Retrieve the leading EOF, expressed as the covariance between the leading PC
# time series and the input SLP anomalies at each grid point.
eof_cov = -solver.eofsAsCovariance(neofs=6)[param['eof_mode'], :, :]
eof_cor = -solver.eofsAsCorrelation(neofs=6)[param['eof_mode'], :, :]
eof = -solver.eofs(neofs=6)[param['eof_mode'], :, :]
pc = -solver.pcs(npcs=6, pcscaling=1)[:, param['eof_mode']]
var = solver.varianceFraction(neigs=6)[param['eof_mode']]
eigs = solver.eigenvalues()[:10]

pc_name = 'PC' + str(param['eof_mode']+1)
eof_name = 'EOF' + str(param['eof_mode']+1) + ' correlation'
var_str = ' (' + str(np.round(var.values*100, 1)) + '%)'

##############################################################################
# PLOT                                                                       #
##############################################################################

f = plt.figure(figsize=(20, 20))
gs = GridSpec(2, 2, figure=f, width_ratios=[0.99, 0.01], height_ratios=[2, 1],
              hspace=0.12, wspace=0.02)
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())) # Main figure (eof)
ax.append(f.add_subplot(gs[0, 1])) # Colorbar for main figure
ax.append(inset_axes(ax[0], width='30%', height='20%', loc=4, borderpad=3))
ax.append(f.add_subplot(gs[1, :])) # pc Clim

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='white',
                                        facecolor='black',
                                        zorder=1)

ax[0].set_aspect(1)

# Plot EOF
eof_max = np.round(np.max(np.abs(eof)), 3)
eof_plot = ax[0].contourf(fmatrix.coords['longitude'], fmatrix.coords['latitude'], eof_cor.T,
                           cmap=cmr.fusion_r, transform=ccrs.PlateCarree(), extend='neither',
                           levels=np.linspace(-1, 1, 21), rasterized=True)
ax[0].add_feature(land_10m)
gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=0.5, color='white', linestyle='--', zorder=11)
gl.xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl.xlabels_top = False
gl.ylabels_right = False
gl.ylabel_style = {'size': 24}
gl.xlabel_style = {'size': 24}
ax[0].text(21, -39, eof_name + var_str, fontsize=32, color='k', fontweight='bold', zorder=20, ha='left', va='bottom')
if param['time'] == 'sink':
    ax[0].set_title(param['name'] + ' (Sink time)', fontsize=32, color='k', fontweight='bold')
else:
    ax[0].set_title(param['name'] + ' (Source time)', fontsize=32, color='k', fontweight='bold')
ax[0].set_xlim([20, 130])
ax[0].set_ylim([-40, 30])

cb0 = plt.colorbar(eof_plot, cax=ax[1], fraction=0.1)
ax[1].tick_params(axis='y', labelsize=22)
cb0.set_label('Correlation', size=24)
cb0.set_ticks(np.linspace(-1, 1, 5))

# Plot PC
yr0 = 1995 if param['time'] == 'sink' else 1993
t_axis = np.arange(yr0, 2013, 1/12) + 1/24
ax[3].plot(t_axis, pc, 'k--', label=pc_name)

ax[3].spines['top'].set_visible(False)
ax[3].spines['right'].set_visible(False)

# ax[3].set_ylabel('Value', fontsize=22)
if param['time'] == 'sink':
    ax[3].set_xlabel('Sink (beaching) year', fontsize=22)
else:
    ax[3].set_xlabel('Source (generation) year', fontsize=22)
ax[3].set_xticks(np.arange(yr0, 2015, 2))
ax[3].tick_params(axis='x', labelsize=22)
ax[3].tick_params(axis='y', labelsize=22)
ax[3].grid(axis='x')
ax[3].set_xlim([yr0, 2013])
ax[3].set_ylim([-np.max(np.abs(pc)), np.max(np.abs(pc))])

# Calculate and add monthly climatology
pc_mc = np.zeros_like(pc)
for month in np.arange(12):
    pc_mc[month::12] = np.mean(pc[month::12])

ax[3].plot(t_axis, pc_mc, 'k-', label=pc_name + ' monthly climatology', alpha=0.5)
ax[3].legend(loc="upper right", frameon=True, fontsize=22, edgecolor='w')

# Add monthly climatology separately for clarity
tm_axis = np.arange(12)
ax[2].plot(tm_axis, pc_mc[:12], 'k-', label=pc_name + ' monthly climatology')

ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].spines['bottom'].set_color('k')
ax[2].spines['left'].set_color('k')

ax[2].set_xticks(np.arange(12))
ax[2].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], color='k')
ax[2].tick_params(axis='x', labelsize=16, color='k')
ax[2].tick_params(axis='y', labelsize=16, color='k')
ax[2].set_xlim([0, 11])
ax[2].set_ylim([-np.max(np.abs(pc_mc[:12])), np.max(np.abs(pc_mc[:12]))])
ax[2].patch.set_facecolor('w')
ax[2].patch.set_alpha(0.3)
ax[2].set_yticks(np.arange(-1, 2, 1))

plt.text(0.093, 0.87, 'a', fontsize=32, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.093, 0.36, 'b', fontsize=32, fontweight='bold', transform=plt.gcf().transFigure)

# Save
plt.savefig(dirs['fig'] + 'EOF_' + param['time'] + '_' + param['physics'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf', bbox_inches='tight', dpi=300)