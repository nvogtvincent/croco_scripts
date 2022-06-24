#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise marine debris results
@author: noam
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cmasher as cmr
import xarray as xr
from osgeo import gdal, osr
from skimage.measure import block_reduce
from matplotlib.gridspec import GridSpec
from sys import argv
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio

# PARAMETERS
param = {'grid_res': 1.0,                          # Grid resolution in degrees
         'lon_range': [20, 130],                   # Longitude range for output
         'lat_range': [-40, 30],                   # Latitude range for output

         # Analysis parameters
         'us_d': argv[1],    # Sinking timescale (days)
         'ub_d': argv[2],      # Beaching timescale (days)

         # Time range
         'y0'  : 1993,
         'y1'  : 2012,

         # Physics
         'mode': argv[3],

         # Source/sink time
         'time': 'source',

         # Sink sites
         'sites': np.array([1,2,3,4]),

         # Names
         'sink': 'Aldabra Group',
         'class': argv[4],

         # Plot ship tracks
         'tracks': True,

         # Export
         'export': True}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'fisheries': os.path.dirname(os.path.realpath(__file__)) + '/../FISHERIES/DATA/PROC/',
        'shipping': os.path.dirname(os.path.realpath(__file__)) + '/../SHIPPING/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
array_str = np.array2string(param['sites'], separator='-').translate({ord(i): None for i in '[]'})
fh = {'flux': dirs['script'] + '/marine_' + param['time'] + '_flux_' + param['mode'] + '_' + array_str + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'debris': dirs['fisheries'] + 'GFW_gridded.nc',
      'shipping': dirs['shipping'] + 'global_shipping.tif'}

n_years = param['y1']-param['y0']+1
##############################################################################
# LOAD DATA                                                                  #
##############################################################################

fmatrix = xr.open_dataarray(fh['flux'])

# To find the proportion of released debris that arrived at site, divide the
# total flux from a cell by the total released, which is (36*4*12) per year.
if param['time'] == 'sink':
    fmatrix_mean = fmatrix.sum(dim='sink_time')/(1728*n_years)
else:
    fmatrix_mean = fmatrix.sum(dim='source_time')/(1728*n_years)

# Now degrade resolution by factor 12
fmatrix_mean_12 = block_reduce(fmatrix_mean, block_size=(12, 12), func=np.mean)

# Open debris input functions (from fisheries)
debris = xr.open_dataset(fh['debris'])

# Open ship tracks
with rasterio.open(fh['shipping']) as src:
    sf = 10
    img = block_reduce(src.read(1), block_size=(sf, sf), func=np.sum)
    height = img.shape[0]
    width = img.shape[1]
    cols, rows = np.meshgrid(np.arange(0, width*10, sf), np.arange(0, height*10, sf))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    lons= np.array(xs)
    lats = np.array(ys)

    # Apply gaussian filter to improve contourf quality
    img = gaussian_filter(img, sigma=1)

# img = block_reduce(img[:10520, :15700], (20, 20), np.sum)
# lons = lons[:10520, :15700][::20, ::20]
# lats = lats[:10520, :15700][::20, ::20]

##############################################################################
# PLOT                                                                       #
##############################################################################

f = plt.figure(constrained_layout=True, figsize=(23.6, 10))
gs = GridSpec(3, 4, figure=f, width_ratios=[0.91, 0.01, 0.29, 0.01])
ax = []
ax.append(f.add_subplot(gs[:, 0], projection = ccrs.PlateCarree())) # Main figure (flux probability)
ax.append(f.add_subplot(gs[:, 1])) # Colorbar for main figure
ax.append(f.add_subplot(gs[0, 2], projection = ccrs.PlateCarree())) # Fisheries 1
ax.append(f.add_subplot(gs[1, 2], projection = ccrs.PlateCarree())) # Fisheries 2
ax.append(f.add_subplot(gs[2, 2], projection = ccrs.PlateCarree())) # Fisheries 3
ax.append(f.add_subplot(gs[:, 3])) # Colorbar for fisheries

# f.subplots_adjust(hspace=0.02, wspace=0.02)
gl = []
hist = []

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='white',
                                        facecolor='black',
                                        zorder=1)

for i in [0, 2, 3, 4]:
    ax[i].set_aspect(1)
    ax[i].set_facecolor('k')

# Total flux
hist.append(ax[0].pcolormesh(fmatrix.coords['longitude'], fmatrix.coords['latitude'], fmatrix_mean,
                             cmap=cmr.ember, norm=colors.LogNorm(vmin=1e-5, vmax=1e-2),
                             transform=ccrs.PlateCarree(), rasterized=True))

ax[0].add_feature(land_10m)
gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='white', linestyle='--', zorder=11))
gl[0].xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[0].xlabels_top = False
gl[0].ylabels_right = False
gl[0].ylabel_style = {'size': 18}
gl[0].xlabel_style = {'size': 18}
ax[0].text(21, -39, 'Likelihood of ' + param['class'] + ' debris beaching at ' + param['sink'], fontsize=28, color='w', fontweight='bold')
ax[0].set_xlim([20, 130])
ax[0].set_ylim([-40, 30])
# Add an overlay with ship tracks
if param['tracks']:
    thresh = 1e7
    img = np.ma.masked_where(img < thresh, img)
    sig_plot = ax[0].contourf(lons, lats, img,
                              levels=np.array([thresh, np.max(img)*2]), hatches=['\\\\\\\\'], colors='none')


cb0 = plt.colorbar(hist[0], cax=ax[1], fraction=0.1)
cb0.set_label('Mass fraction', size=24)
ax[1].tick_params(axis='y', labelsize=22)

# Input from purse seiners
ps_input = fmatrix_mean_12*debris['purse_seiners_B_time_integral']
hist.append(ax[2].pcolormesh(debris.coords['lon_bnd'], debris.coords['lat_bnd'],
                             ps_input/ps_input.sum(),
                             cmap=cmr.ghostlight, norm=colors.LogNorm(vmin=1e-4, vmax=1e-2),
                             transform=ccrs.PlateCarree(), rasterized=True))
ax[2].add_feature(land_10m)
ax[2].text(21, -39, 'Purse seiners', fontsize=22, color='w', fontweight='bold')

# Input from drifting and set longlines
ll_input = fmatrix_mean_12*debris['longlines_B_time_integral']
hist.append(ax[3].pcolormesh(debris.coords['lon_bnd'], debris.coords['lat_bnd'],
                             ll_input/ll_input.sum(),
                             cmap=cmr.ghostlight, norm=colors.LogNorm(vmin=1e-4, vmax=1e-2),
                             transform=ccrs.PlateCarree(), rasterized=True))
ax[3].add_feature(land_10m)
ax[3].text(21, -39, 'Set/drifting longlines', fontsize=22, color='w', fontweight='bold')

# Input from pole and line and trollers
pl_input = fmatrix_mean_12*debris['pole_and_line_and_trollers_B_time_integral']
hist.append(ax[4].pcolormesh(debris.coords['lon_bnd'], debris.coords['lat_bnd'],
                             pl_input/pl_input.sum(),
                             cmap=cmr.ghostlight, norm=colors.LogNorm(vmin=1e-4, vmax=1e-2),
                             transform=ccrs.PlateCarree(), rasterized=True))
ax[4].add_feature(land_10m)
ax[4].text(21, -39, 'Pole and line/trollers', fontsize=22, color='w', fontweight='bold')

cb1 = plt.colorbar(hist[1], cax=ax[5])
cb1.set_label('Normalised risk from fishery', size=24)
ax[5].tick_params(axis='y', labelsize=22)

for ii, i in enumerate([2, 3, 4]):
    gl.append(ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='white', linestyle='--', zorder=11))
    gl[ii+1].xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
    gl[ii+1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[ii+1].xlabels_top = False
    gl[ii+1].ylabels_right = False
    gl[ii+1].ylabels_left = False
    gl[ii+1].xlabels_bottom = False

if param['export']:
    plt.savefig(dirs['fig'] + 'marine_sources_' + param['mode'] + '_' + np.array2string(param['sites'], separator='-') + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.png', bbox_inches='tight', dpi=300)








