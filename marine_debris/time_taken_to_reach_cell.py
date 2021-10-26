#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates statistics on the time taken for particles to reach a
particular cell
@author: Noam Vogt-Vincent


Methodology:
1. Set up a binning grid
2. For a certain number of trajectories, find the grid indices for each point
"""

import psutil
import os
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import numpy as np
import cmocean.cm as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time as pytime
from netCDF4 import Dataset
from glob import glob


##############################################################################
# Parameters #################################################################
##############################################################################

# PARAMETERS
param = {'res'       :         0.5,             # Binning resolution (degrees)
         'lon_range' : [-180, 180],             # Longitude range [min, max]
         'lat_range' : [-90 , 90 ],             # Latitude range  [min, max]
         'out_name'  : 'time2reach.png',}       # Output figure name

# DIRECTORIES
dirs = {'script'     : os.path.dirname(os.path.realpath(__file__)),
        'grid'       : os.path.dirname(os.path.realpath(__file__)) + '/GRID_DATA/',
        'traj'       : os.path.dirname(os.path.realpath(__file__)) + '/TRAJ/',
        'fig'        : os.path.dirname(os.path.realpath(__file__)) + '/FIGURES/'}

# FILE HANDLES
fh = {'grid'         : dirs['grid'] + 'griddata.nc',
      'traj'         : sorted(glob(dirs['traj'] + '*.nc')),
      'fig'          : dirs['fig'] + param['out_name']}

##############################################################################
# Binning ####################################################################
##############################################################################

# Generate a global grid
lon_bnd = np.linspace(param['lon_range'][0],
                      param['lon_range'][1],
                      int((param['lon_range'][1] - param['lon_range'][0])/param['res'])+1)

lat_bnd = np.linspace(param['lat_range'][0],
                      param['lat_range'][1],
                      int((param['lat_range'][1] - param['lat_range'][0])/param['res'])+1)

min_days = np.ones((len(lat_bnd)-1, len(lon_bnd)-1),
                   dtype=np.int16)*32767

# Find the dimensions of the trajectory data
param['nParticleFile'] = len(fh['traj'])

@njit
def update_min_time(lon_full, lat_full, time_full, grid, data):
    lat_bnd = grid[0]
    lon_bnd = grid[1]

    ntraj = lon_full.shape[0]
    for i in range(ntraj):
        # Load positions and times for this trajectory
        lon = lon_full[i, :]
        lat = lat_full[i, :]
        time = time_full[i, :]

        mask = ~np.isnan(lon)
        lon = lon[mask]
        lat = lat[mask]
        time = time[mask]

        # Find indices on grid for all particle positions
        lat_idx = np.searchsorted(lat_bnd, lat)-1
        lon_idx = np.searchsorted(lon_bnd, lon)-1

        for j in range(len(lon)):
            # Check if time in current grid bin is less than the previous min
            # If so, overwrite
            old_val = data[lat_idx[j], lon_idx[j]]
            if time[j] < old_val:
                data[lat_idx[j], lon_idx[j]] = time[j]

    return data


for pf in range(param['nParticleFile']):
    with Dataset(fh['traj'][pf], mode='r') as nc:
        # Firstly calculate data chunking
        ntime = nc.variables['trajectory'].shape[1]
        ntraj = nc.variables['trajectory'].shape[0]

        avail_mem = psutil.virtual_memory()[4]
        c_ntraj   = np.ceil(avail_mem/(ntime*8*10)).astype('int') # Trajectories per chunk
        c_n       = np.ceil(ntraj/c_ntraj).astype('int')          # Number of chunks

        for i in range(c_n): #c_n
            # Load in chunk data
            c_lon  = nc.variables['lon'][i*c_ntraj:(i+1)*c_ntraj]
            c_lat  = nc.variables['lat'][i*c_ntraj:(i+1)*c_ntraj]
            c_time = nc.variables['time'][i*c_ntraj:(i+1)*c_ntraj]

            c_time = c_time - np.repeat(c_time[:, 0], c_time.shape[1]).reshape((-1, c_time.shape[1]))
            c_time = np.abs(c_time)/86400. # Convert to days at sea

            # Now update min_days
            # start = pytime.time()

            min_days = update_min_time(np.array(c_lon),
                                       np.array(c_lat),
                                       np.array(c_time).astype(np.int16),
                                       (lat_bnd, lon_bnd),
                                       min_days)
            # print(pytime.time()-start)

# min_days = np.ma.masked_equal(min_days, 32767)/365
min_days = min_days/365

##############################################################################
# Plot #######################################################################
##############################################################################

f, a0 = plt.subplots(1, 1, figsize=(20, 10),
                     subplot_kw={'projection': ccrs.PlateCarree()})
f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)

# Set up the colorbar
axpos = a0.get_position()
pos_x = axpos.x0+axpos.width + 0.22
pos_y = axpos.y0
cax_width = 0.02
cax_height = axpos.height

pos_cax = f.add_axes([pos_x, pos_y, cax_width, cax_height])

# Set up the colormap (from https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar)
cmap = cmo.thermal_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = colors.LinearSegmentedColormap.from_list('dense_sq', cmaplist, cmap.N)
cmap_bounds = np.linspace(0, 10, 11)
cmap_norm = colors.BoundaryNorm(cmap_bounds, cmap.N)

# Plot the colormesh
hist = a0.pcolormesh(lon_bnd, lat_bnd, min_days, cmap=cmap, norm=cmap_norm,
                     vmin=1, vmax=10)

# Add cartographic features
gl = a0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='black', linestyle='-', zorder=11)
gl.xlabels_top = False
gl.ylabels_right = False

a0.add_feature(cfeature.LAND, facecolor='white', zorder=10)

cb = plt.colorbar(hist, cax=pos_cax)
cb.set_label('Minimum years at sea to Seychelles', size=12)
a0.set_aspect('auto', adjustable=None)
a0.margins(x=-0.01, y=-0.01)
plt.savefig(fh['fig'], dpi=300)

print('...complete!')
