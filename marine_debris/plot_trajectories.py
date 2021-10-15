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
grid_file = script_dir + '/GRID_DATA/griddata.nc'

traj_file1 = script_dir + '/TRAJ/incorrect_wave.nc'

output_name = '/trajectories.png'

##############################################################################
# Load bathymetry data #######################################################
##############################################################################

with Dataset(grid_file, mode='r') as nc:
    lon = np.array(nc.variables['lon_rho'][:])
    lat = np.array(nc.variables['lat_rho'][:])
    lon_c = np.array(nc.variables['lon_psi'][:])
    lat_c = np.array(nc.variables['lat_psi'][:])
    mask = np.array(nc.variables['lsm_rho'][:])

mask = np.ma.masked_where(mask == 1, mask)

##############################################################################
# Load trajectory data #######################################################
##############################################################################

with Dataset(traj_file1, mode='r') as nc:
    plon1 = np.array(nc.variables['lon'][:])
    plat1 = np.array(nc.variables['lat'][:])
    pt1   = np.array(nc.variables['time'][:])

jidx = np.argmax(pt1 == 43200, axis=1)
iidx = np.arange(0, len(jidx), 1)

last_lon = np.take_along_axis(plon1, jidx.reshape(-1,1), axis=1)
last_lat = np.take_along_axis(plat1, jidx.reshape(-1,1), axis=1)

last_lon_fail = np.ma.masked_where(jidx.reshape(-1,1) != 0, last_lon)
last_lat_fail = np.ma.masked_where(jidx.reshape(-1,1) != 0, last_lat)

last_lon_succ = np.ma.masked_where(jidx.reshape(-1,1) == 0, last_lon)
last_lat_succ = np.ma.masked_where(jidx.reshape(-1,1) == 0, last_lat)

##############################################################################
# Plot #######################################################################
##############################################################################

f, a0 = plt.subplots(1, 1, figsize=(20, 10))
f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)

lsm = a0.pcolormesh(lon, lat, 1-mask, cmap=cmo.gray)

# gl = a0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=1, color='black', alpha=0.5, linestyle='--')
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.ylocator = mticker.FixedLocator(np.linspace(-25,0,6))
# gl.xlocator = mticker.FixedLocator(np.linspace(30,80,11))

n_traj = np.shape(plon1)[0]

for i in range(n_traj):
    a0.plot(plon1[i, :], plat1[i, :], linewidth=0.5, color='white', alpha=0.02)
    print(i)

a0.scatter(last_lon_fail, last_lat_fail, s=2, c='r')
a0.scatter(last_lon_succ, last_lat_succ, s=2, c='b')

# for i in range(n_traj):
#     a0.plot(plon2[i, :], plat2[i, :], linewidth=0.5, color='orange', alpha=0.5)

# for i in range(n_traj):
#     a0.plot(plon3[i, :], plat3[i, :], linewidth=0.5, color='gray', alpha=0.5)

a0.set_xlim(0, 180)
a0.set_ylim(-50, 40)
#a0.set_aspect('auto', adjustable=None)
a0.margins(x=-0.01, y=-0.01)
plt.savefig(script_dir + output_name, dpi=300)

print('...complete!')
