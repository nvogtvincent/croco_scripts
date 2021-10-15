#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 23:05:50 2021
This script plots selected particle directories
@author: noam
"""

import psutil
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import numpy as np
import cmocean.cm as cmo
import cartopy.crs as ccrs
import random
from netCDF4 import Dataset

##############################################################################
# Parameters #################################################################
##############################################################################

script_dir = os.path.dirname(os.path.realpath(__file__))
grid_file = script_dir + '/source_files/CMEMS_GRID.nc'
traj_file = script_dir + '/source_files/1M_2019bwd_delete.nc'

output_name = '/trajectories.png'

res = 0.5  # Resolution for binning (degrees)
lon_range = [-180, 180]
lat_range = [-90, 90]

##############################################################################
# Load bathymetry data #######################################################
##############################################################################

# with Dataset(grid_file, mode='r') as nc:
#     lon = np.array(nc.variables['longitude'][:])
#     lat = np.array(nc.variables['latitude'][:])
#     lon_c = np.array(nc.variables['lon_psi'][:])
#     lat_c = np.array(nc.variables['lat_psi'][:])
#     h = np.array(nc.variables['h'][:])
#     mask = np.array(nc.variables['mask_rho'][:])
#     coral = np.array(nc.variables['reef_loc'][:])

# mask = np.ma.masked_where(mask == 1, mask)
# coral = np.ma.masked_where(coral == 0, coral)

# Generate a global grid
lon_bnd = np.linspace(lon_range[0],
                      lon_range[1],
                      int((lon_range[1] - lon_range[0])/res)+1)

lat_bnd = np.linspace(lat_range[0],
                      lat_range[1],
                      int((lat_range[1] - lat_range[0])/res)+1)

##############################################################################
# Load trajectory data #######################################################
##############################################################################

# Detect available system memory
avail_mem = psutil.virtual_memory()[4]

with Dataset(traj_file, mode='r') as nc:
    ntraj, ntime = np.shape(nc.variables['lon'])
    buff = 4
    exp_size = ntraj*ntime*8*2*buff # Maximum size of both lon + lat + buffer

    if exp_size > avail_mem:
        # If insufficient memory, split histograms into chunks and add them
        # together

        # Maximum number of trajectories per chunk
        max_traj = int(avail_mem/(ntime*8*2*buff))

        # Number of chunks required
        nchunk = int(np.ceil(ntraj/max_traj))
        nchunk = 1

        plon = nc.variables['lon'][:max_traj]
        plat = nc.variables['lat'][:max_traj]

        # Find identical final trajectories
        dupe_traj = np.unique(plon[:, 600:], axis=0, return_inverse=True, return_counts=True)
        idx = np.where(dupe_traj[2] == 2)[0][5]
        dupe_traj_id = np.where(dupe_traj[1] == idx)[0]

        dplon = plon[dupe_traj_id, :]
        dplat = plat[dupe_traj_id, :]

# Find confluence
i = 0
while i >= 0:
    match = (dplon[0, i] == dplon[1, i])
    if match:
        conf_lon = dplon[0, i]
        conf_lat = dplat[0, i]
        i = -1
    else:
        i += 1


##############################################################################
# Plot #######################################################################
##############################################################################

f, a0 = plt.subplots(1, 1, figsize=(20, 10),
                     subplot_kw={'projection': ccrs.PlateCarree()})
f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)

for i in range(np.shape(dplon)[0]):
    a0.plot(dplon[i, :], dplat[i, :], linewidth=1,
            c=('b', 'r')[i])

a0.scatter(conf_lon, conf_lat, s=50, c='k')

a0.set_extent([-180, 180, -70, 70], crs=ccrs.PlateCarree())

gl = a0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
coastl = a0.coastlines()

a0.set_aspect('auto', adjustable=None)
a0.margins(x=-0.01, y=-0.01)
plt.savefig(script_dir + output_name, dpi=300)

print('...complete!')
