#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 23:05:50 2021
This script plots selected particle directories
@author: noam
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from netCDF4 import Dataset
import numpy as np
import cmocean.cm as cmo
import cartopy.crs as ccrs
import os

##############################################################################
# Parameters #################################################################
##############################################################################

script_dir = os.path.dirname(os.path.realpath(__file__))
grid_file = script_dir + '/source_files/croco_grd.nc'

traj_file1 = script_dir + '/source_files/test.nc'
traj_file2 = script_dir + '/source_files/test2.nc'
traj_file3 = script_dir + '/source_files/test3.nc'


output_name = '/trajectories.png'

##############################################################################
# Load bathymetry data #######################################################
##############################################################################

with Dataset(grid_file, mode='r') as nc:
    lon = np.array(nc.variables['lon_rho'][:])
    lat = np.array(nc.variables['lat_rho'][:])
    lon_c = np.array(nc.variables['lon_psi'][:])
    lat_c = np.array(nc.variables['lat_psi'][:])
    h = np.array(nc.variables['h'][:])
    mask = np.array(nc.variables['mask_rho'][:])
    coral = np.array(nc.variables['reef_loc'][:])

mask = np.ma.masked_where(mask == 1, mask)
coral = np.ma.masked_where(coral == 0, coral)

##############################################################################
# Load trajectory data #######################################################
##############################################################################

with Dataset(traj_file1, mode='r') as nc:
    plon1 = np.array(nc.variables['lon'][:])
    plat1 = np.array(nc.variables['lat'][:])

with Dataset(traj_file2, mode='r') as nc:
    plon2 = np.array(nc.variables['lon'][:])
    plat2 = np.array(nc.variables['lat'][:])

with Dataset(traj_file3, mode='r') as nc:
    plon3 = np.array(nc.variables['lon'][:])
    plat3 = np.array(nc.variables['lat'][:])

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

bath = a0.pcolormesh(lon, lat, h/1e3, cmap=cmo.deep)
lsm = a0.pcolormesh(lon, lat, mask, cmap=cmo.gray)
cm = a0.pcolormesh(lon_c, lat_c, coral, cmap=cmo.amp, vmin = 0, vmax = 1.5)


gl = a0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.ylocator = mticker.FixedLocator(np.linspace(-25,0,6))
gl.xlocator = mticker.FixedLocator(np.linspace(30,80,11))

n_traj = np.shape(plon1)[0]

for i in range(n_traj):
    a0.plot(plon1[i, :], plat1[i, :], linewidth=0.5, color='white', alpha=0.5)

# for i in range(n_traj):
#     a0.plot(plon2[i, :], plat2[i, :], linewidth=0.5, color='orange', alpha=0.5)

for i in range(n_traj):
    a0.plot(plon3[i, :], plat3[i, :], linewidth=0.5, color='gray', alpha=0.5)

cb = plt.colorbar(bath, cax=pos_cax)
cb.set_label('Depth (km)', size=12)
a0.set_aspect('auto', adjustable=None)
a0.margins(x=-0.01, y=-0.01)
a0.set_ylim(top=0)
plt.savefig(script_dir + output_name, dpi=300)

print('...complete!')
