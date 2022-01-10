#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script takes backward trajectories from Seychelles and applies a simple
parameterisation for fragmentation-sinking and beaching to estimate the
fraction of mass passing through a grid cell that reaches Seychelles given that
the mass is on a trajectory towards Seychelles.

Parameters used:
    lb: rate constant for mass loss due to beaching (1/s)
    ls: rate constant for mass loss due to sinking (1/s)

@author: Noam Vogt-Vincent
"""

import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as ani
import cmasher as cmr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from netCDF4 import Dataset
from numba import njit
from psutil import virtual_memory
from glob import glob


###############################################################################
# Parameters ##################################################################
###############################################################################

# PARAMETERS
param = {'grid_res': 1,                          # Grid resolution in degrees
         'lon_range': [-180, 180],                 # Longitude range for output
         'lat_range': [-90, 90],                   # Latitude range for output

         'lb': 60,                                 # Days
         'ls': 3650,                               # Days

         'days_per_obs': 5,                        # Days represented by each obs
         'percentile': 90,}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)) + '/../',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/../TRAJ/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'traj': sorted(glob(dirs['traj'] + '*SeyBwd.nc')),
      'fig': dirs['fig'] + 'b' + str(param['lb']) +
      '_s' + str(param['ls']) + '_p' + str(param['percentile'])}

red_mem = 100 # (workaround for HPC)
param['lb'] = 1/(3600*24*param['lb'])
param['ls'] = 1/(3600*24*param['ls'])
###############################################################################
# Routines ####################################################################
###############################################################################


def get_init_times(param, fh):
    # Get a list of times for each month
    for fh_num, traj_fh in enumerate(fh['traj']):
        with Dataset(fh['traj'][fh_num], mode='r') as nc:
            ch_times = np.unique(nc['time'][:, 0])

            if fh_num == 0:
                init_times = ch_times
            else:
                init_times = np.concatenate((init_times, ch_times))

    # Ensure list is sorted
    init_times = np.sort(init_times)

    # Organise by month
    init_times = init_times.reshape((-1, 12))

    return init_times


def get_traj(fh, fh_num, start, end, **kwargs):
    with Dataset(fh['traj'][fh_num], mode='r') as nc:
        lon_out = nc.variables['lon'][start:end, :]
        lat_out = nc.variables['lat'][start:end, :]
        ts_out = nc.variables['time'][start:end, :]
        tb_out = nc.variables['ct'][start:end, :]

        if 'initial' in kwargs:
            mask = np.isin(ts_out[:, 0], kwargs['initial'])
            lon_out = lon_out[mask]
            lat_out = lat_out[mask]
            ts_out = ts_out[mask]

        # Convert ts to time at sea
        ts_out = ts_out[:, 0].reshape(-1, 1) - ts_out

        # Calculate number of trajectories
        ntraj = len(ts_out[:, 0])

    return [ntraj,
            lon_out.compressed(),
            lat_out.compressed(),
            ts_out.compressed(),
            tb_out.compressed()]


def calc_loss(lon_bnd, lat_bnd, mass_bnd, ot_bnd, ct_bnd, param, fh):
    # Calculate available memory
    avail_mem = virtual_memory()[4]

    # Generate the empty grids
    hist_grid_massloss = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1, len(mass_bnd)-1), dtype=np.int32)
    hist_grid_ot = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1, len(ot_bnd)-1), dtype=np.int32)
    hist_grid_ct = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1, len(ct_bnd)-1), dtype=np.int32)

    grid_massloss = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1), dtype=np.float32)
    grid_ot = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1), dtype=np.float32)
    grid_ct = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1), dtype=np.float32)

    # Loop over trajectory files
    for fh_num, traj_fh in enumerate(fh['traj']):
        # Calculate chunking for this trajectory file
        with Dataset(fh['traj'][0], mode='r') as nc:
            ntraj, ntime = np.shape(nc.variables['lon'])
            ntraj_ch = int(avail_mem/(ntime*8*red_mem))
            n_ch = int(np.ceil(ntraj/ntraj_ch))

        # Loop over chunks
        for chunk in range(n_ch):
            # Load data
            data = get_traj(fh, fh_num,
                            ntraj_ch*chunk,
                            ntraj_ch*(chunk+1))[1:]

            # Calculate mass ratio at each point
            mass = np.exp((-param['lb']*data[3])+(-param['ls']*data[2]))

            ot = data[2]/(3600*24*365.25) # Days
            ct = data[3]/(3600*24*365.25) # Days

            hist_grid_massloss += np.histogramdd(np.array([data[1], data[0], mass]).T,
                                                 (lat_bnd, lon_bnd, mass_bnd))[0].astype(np.int32)
            hist_grid_ot += np.histogramdd(np.array([data[1], data[0], ot]).T,
                                           (lat_bnd, lon_bnd, ot_bnd))[0].astype(np.int32)
            hist_grid_ct += np.histogramdd(np.array([data[1], data[0], ct]).T,
                                           (lat_bnd, lon_bnd, ct_bnd))[0].astype(np.int32)

    # Finally, calculate the specified percentile
    def find_percentile(lon_bnd, lat_bnd, val_bnd, grid_in, percentile):
        val_mp = 0.5*(val_bnd[1:] + val_bnd[:-1])
        grid = np.ones(np.shape(grid_in)[:2])

        for lat_idx in range(len(lat_bnd)-1):
            for lon_idx in range(len(lon_bnd)-1):
                hist = grid_in[lat_idx, lon_idx, :]
                if np.sum(hist) > 0:
                    grid[lat_idx, lon_idx] = np.percentile(np.repeat(val_mp, hist), percentile)
                else:
                    grid[lat_idx, lon_idx] = -1

        return np.ma.masked_where(grid < 0, grid)

    grid_massloss = find_percentile(lon_bnd, lat_bnd, mass_bnd, hist_grid_massloss, param['percentile'])
    grid_ot = find_percentile(lon_bnd, lat_bnd, ot_bnd, hist_grid_ot, 100-param['percentile'])
    grid_ct = find_percentile(lon_bnd, lat_bnd, ct_bnd, hist_grid_ct, 100-param['percentile'])

    return grid_massloss, grid_ot, grid_ct


###############################################################################
# Run script for current parameters ###########################################
###############################################################################

if __name__ == '__main__':
    # Generate a global grid
    lon_bnd = np.linspace(param['lon_range'][0],
                          param['lon_range'][1],
                          int((param['lon_range'][1] -
                               param['lon_range'][0])/param['grid_res'])+1)

    lat_bnd = np.linspace(param['lat_range'][0],
                          param['lat_range'][1],
                          int((param['lat_range'][1] -
                               param['lat_range'][0])/param['grid_res'])+1)

    mass_bnd = np.linspace(0, 1, num=101)
    ot_bnd = np.linspace(0, 12, num=145)
    ct_bnd = np.linspace(0, 1, num=101)
    ot_bnd[-1] = 30
    ct_bnd[-1] = 30

    print('Gridding trajectories...')

    ###########################################################################
    # Plotting ################################################################
    ###########################################################################

    # Calculate losses from beaching and sinking
    grids = calc_loss(lon_bnd, lat_bnd, mass_bnd, ot_bnd, ct_bnd, param, fh)

    print('Calculation complete')


    # Plotting loss
    f, ax = plt.subplots(1, 1, figsize=(19, 10),
                         subplot_kw={'projection': ccrs.Robinson(central_longitude=60)})

    f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    ax.set_global()
    ax.set_aspect(1)

    # Set up the colorbar
    pos_cax = f.add_axes([ax.get_position().x1+0.01,ax.get_position().y0-0.025,0.015,ax.get_position().height+0.05])

    # Plot the colormesh
    cmap = cmr.fall_r
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # cmap = colors.LinearSegmentedColormap.from_list('dense_sq', cmaplist, cmap.N)
    # cmap_bounds = np.linspace(0, 1, 11)
    # cmap_norm = colors.BoundaryNorm(cmap_bounds, cmap.N)

    hist = ax.pcolormesh(lon_bnd, lat_bnd, grids[0], cmap=cmap, vmin=0, vmax=1,
                         transform=ccrs.PlateCarree())

    # Add cartographic features
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='black', linestyle='--', zorder=11)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 240, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 30))
    gl.xlabels_top = False
    gl.ylabels_right = False

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='gray')
    ax.add_feature(land_10m)

    cb = plt.colorbar(hist, cax=pos_cax)
    cb.set_label('Mass fraction reaching Seychelles', size=12)
    ax.set_aspect('auto', adjustable=None)
    ax.margins(x=-0.01, y=-0.01)
    plt.savefig(fh['fig'] + '_loss.png', dpi=300)
    plt.close()


    # Plotting ocean time
    f, ax = plt.subplots(1, 1, figsize=(19, 10),
                         subplot_kw={'projection': ccrs.Robinson(central_longitude=60)})

    f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    ax.set_global()
    ax.set_aspect(1)

    # Set up the colorbar
    pos_cax = f.add_axes([ax.get_position().x1+0.01,ax.get_position().y0-0.025,0.015,ax.get_position().height+0.05])

    # Plot the colormesh
    cmap = cmr.ocean
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list('dense_sq', cmaplist, cmap.N)
    cmap_bounds = np.linspace(0, 12, 13)
    cmap_norm = colors.BoundaryNorm(cmap_bounds, cmap.N)

    hist = ax.pcolormesh(lon_bnd, lat_bnd, grids[1], cmap=cmap, vmin=0, vmax=12,
                         norm=cmap_norm, transform=ccrs.PlateCarree())

    # Add cartographic features
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='black', linestyle='--', zorder=11)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 240, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 30))
    gl.xlabels_top = False
    gl.ylabels_right = False

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='gray')
    ax.add_feature(land_10m)

    cb = plt.colorbar(hist, cax=pos_cax)
    cb.set_label('Time at sea to Seychelles (years, 10th percentile)', size=12)
    ax.set_aspect('auto', adjustable=None)
    ax.margins(x=-0.01, y=-0.01)
    plt.savefig(fh['fig'] + '_ot.png', dpi=300)
    plt.close()

    # Plotting ocean time
    f, ax = plt.subplots(1, 1, figsize=(19, 10),
                         subplot_kw={'projection': ccrs.Robinson(central_longitude=60)})

    f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    ax.set_global()
    ax.set_aspect(1)

    # Set up the colorbar
    pos_cax = f.add_axes([ax.get_position().x1+0.01,ax.get_position().y0-0.025,0.015,ax.get_position().height+0.05])

    # Plot the colormesh
    cmap = cmr.swamp_r
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list('dense_sq', cmaplist, cmap.N)
    cmap_bounds = np.linspace(0, 12, 13)
    cmap_norm = colors.BoundaryNorm(cmap_bounds, cmap.N)

    hist = ax.pcolormesh(lon_bnd, lat_bnd, grids[2]*12, cmap=cmap, vmin=0, vmax=12,
                         norm=cmap_norm, transform=ccrs.PlateCarree())

    # Add cartographic features
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='black', linestyle='--', zorder=11)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 240, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 30))
    gl.xlabels_top = False
    gl.ylabels_right = False

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='gray')
    ax.add_feature(land_10m)

    cb = plt.colorbar(hist, cax=pos_cax)
    cb.set_label('Time along the coast enroute to Seychelles (months, 10th percentile)', size=12)
    ax.set_aspect('auto', adjustable=None)
    ax.margins(x=-0.01, y=-0.01)
    plt.savefig(fh['fig'] + '_ct.png', dpi=300)
    plt.close()
