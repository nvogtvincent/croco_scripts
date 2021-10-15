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
        # nchunk = 10

        for c in range(nchunk):
            plon = nc.variables['lon'][c*max_traj:(c+1)*max_traj].compressed()
            plat = nc.variables['lat'][c*max_traj:(c+1)*max_traj].compressed()

            if c == 0:
                p_hist = np.histogram2d(plon, plat, bins=[lon_bnd, lat_bnd])[0]
            else:
                p_hist += np.histogram2d(plon, plat, bins=[lon_bnd, lat_bnd])[0]
    else:
        plon = nc.variables['lon'][:].compressed()
        plat = nc.variables['lat'][:].compressed()

        p_hist = np.histogram2d(plon, plat, bins=[lat_bnd, lon_bnd])[0]

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

# bath = a0.pcolormesh(lon, lat, h/1e3, cmap=cmo.deep)
# lsm = a0.pcolormesh(lon, lat, mask, cmap=cmo.gray)
# cm = a0.pcolormesh(lon_c, lat_c, coral, cmap=cmo.amp, vmin = 0, vmax = 1.5)

hist = a0.pcolormesh(lon_bnd, lat_bnd, p_hist.T, cmap=cmo.thermal,
                     norm=colors.LogNorm(vmin=1, vmax=np.max(p_hist)))

gl = a0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
coastl = a0.coastlines()
# gl.ylocator = mticker.FixedLocator(np.linspace(-25,0,6))
# gl.xlocator = mticker.FixedLocator(np.linspace(30,80,11))

# n_traj = np.shape(plon)[0]

# for i in range(n_traj):
#     a0.plot(plon[i, :], plat[i, :], linewidth=0.5, color='white', alpha=0.5)
#     print(i/n_traj)

cb = plt.colorbar(hist, cax=pos_cax)
cb.set_label('Number of observations', size=12)
a0.set_aspect('auto', adjustable=None)
a0.margins(x=-0.01, y=-0.01)
# a0.set_ylim(top=0)
plt.savefig(script_dir + output_name, dpi=300)

print('...complete!')
