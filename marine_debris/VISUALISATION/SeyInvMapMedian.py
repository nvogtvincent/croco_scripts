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
from netCDF4 import Dataset
from numba import njit
from psutil import virtual_memory
from glob import glob


###############################################################################
# Parameters ##################################################################
###############################################################################

# PARAMETERS
param = {'grid_res': 1.0,                          # Grid resolution in degrees
         'lon_range': [-180, 180],                 # Longitude range for output
         'lat_range': [-90, 90],                   # Latitude range for output

         'lb': 1/(3600*24*60),
         'ls': 1/(3600*24*3650),

         'days_per_obs': 5,                        # Days represented by each obs
         'percentile': 50,

         'out_name_1': 'loss_map.png',
         'out_name_2': 'dens_map_annual_mean.png',
         'out_name_3': 'dens_map_seasons'}     # Output figure name

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)) + '/../',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/../TRAJ/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'traj': sorted(glob(dirs['traj'] + '*.nc')),
      'fig': [dirs['fig'] + param['out_name_1'],
              dirs['fig'] + param['out_name_2'],
              dirs['fig'] + param['out_name_3']]}

red_mem = 100 # (workaround for HPC)

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


def calc_loss(lon_bnd, lat_bnd, mass_bnd, param, fh):
    # Calculate available memory
    avail_mem = virtual_memory()[4]

    # Generate the empty grids
    # tot_grid = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1, 120), dtype=np.float32)
    hist_grid = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1, 100), dtype=np.int32)
    grid = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1), dtype=np.float32)

    # Loop over trajectory files
    for fh_num, traj_fh in enumerate(fh['traj']):
        # Calculate chunking for this trajectory file
        with Dataset(fh['traj'][0], mode='r') as nc:
            ntraj, ntime = np.shape(nc.variables['lon'])
            ntraj_ch = int(avail_mem/(ntime*8*red_mem))
            n_ch = int(np.ceil(ntraj/ntraj_ch))

        # Loop over chunks
        for chunk in range(n_ch): # Change to n_ch
            # Load data
            data = get_traj(fh, fh_num,
                            ntraj_ch*chunk,
                            ntraj_ch*(chunk+1))[1:]

            # Calculate mass ratio at each point
            mass = np.exp((-param['lb']*data[3])+(-param['ls']*data[2]))
            hist_grid += np.histogramdd(np.array([data[1], data[0], mass]).T,
                                        (lat_bnd, lon_bnd, mass_bnd))[0].astype(np.int32)

    # Finally, calculate the specified percentile
    def find_percentile(lon_bnd, lat_bnd, mass_bnd, grid_in, percentile):
        mass_mp = 0.5*(mass_bnd[1:] + mass_bnd[:-1])
        grid = np.ones(np.shape(grid_in)[:2])

        for lat_idx in range(len(lat_bnd)-1):
            for lon_idx in range(len(lon_bnd)-1):
                hist = grid_in[lat_idx, lon_idx, :]
                if np.sum(hist) > 0:
                    grid[lat_idx, lon_idx] = np.percentile(np.repeat(mass_mp, hist), percentile)
                else:
                    grid[lat_idx, lon_idx] = -1

        return grid

    grid = find_percentile(lon_bnd, lat_bnd, mass_bnd, hist_grid, param['percentile'])
    grid = np.ma.masked_where(grid < 0, grid)

    return grid


def calc_dens(lon_bnd, lat_bnd, init_times, param, fh):
    # Calculate available memory
    avail_mem = virtual_memory()[4]

    # Generate the empty grids
    num_grid = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1), dtype=np.int32)

    # Calculate number of chunks
    with Dataset(fh['traj'][0], mode='r') as nc:
        ntraj, ntime = np.shape(nc.variables['lon'])
        ntraj_ch = int(avail_mem/(ntime*8*red_mem))
        n_ch = int(np.ceil(ntraj/ntraj_ch))

    ntraj_tot = 0

    # Extract rows with correct time
    for fh_num, traj_fh in enumerate(fh['traj']):
        for chunk in range(n_ch): # Change to n_ch
            # Load data
            data = get_traj(fh, fh_num,
                            ntraj_ch*chunk,
                            ntraj_ch*(chunk+1),
                            initial=init_times)[:3]

            # Grid
            num_grid += np.histogram2d(data[2], data[1],
                                       (lat_bnd, lon_bnd))[0].astype(np.int32)*param['days_per_obs']

            # Keep track of number of particles
            ntraj_tot += data[0]

    # If ntraj_tot == 0, avoid NaN
    if ntraj_tot == 0:
        print('Warning: total trajectories is 0')
        ntraj_tot = 1

    return num_grid/ntraj_tot


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

    print('Gridding trajectories...')

    # Get initial times for all months
    # init_times = get_init_times(param, fh)
    # dens_month_grid = []
    # for month in range(12):
    #     # Grid locations for this month
    #     dens_month_grid.append(calc_dens(lon_bnd, lat_bnd, init_times[:, month], param, fh))

    # dens_month_grid = np.array(dens_month_grid)

    # # Calculate monthly/annual stats
    # annual_mean = np.mean(dens_month_grid, axis=0)
    # seasonal_means = [np.mean(dens_month_grid[0:3, :, :], axis=0),
    #                   np.mean(dens_month_grid[3:6, :, :], axis=0),
    #                   np.mean(dens_month_grid[6:9, :, :], axis=0),
    #                   np.mean(dens_month_grid[9:12, :, :], axis=0)]

    ###########################################################################
    # Plotting ################################################################
    ###########################################################################

    # f0, a0 = plt.subplots(1, 1, figsize=(20, 10),
    #                       subplot_kw={'projection': ccrs.PlateCarree()})
    # f0.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    # a0.set_ylim([-70, 70])

    # axpos = a0.get_position()
    # pos_x = axpos.x0+axpos.width + 0.41
    # pos_y = axpos.y0
    # cax_width = 0.015
    # cax_height = axpos.height

    # pos_cax = f0.add_axes([pos_x, pos_y, cax_width, cax_height])

    # # Set up the colormap (from https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar)
    # cmap = cmr.ocean
    # dens = a0.pcolormesh(lon_bnd, lat_bnd, annual_mean, cmap=cmap,
    #                      norm=colors.LogNorm(vmin=1e-5, vmax=1e0))

    # # Add cartographic features
    # gl = a0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                   linewidth=0.5, color='black', linestyle='-', zorder=11)
    # gl.xlabels_top = False
    # gl.ylabels_right = False

    # land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
    #                                         edgecolor='face',
    #                                         facecolor='black')
    # a0.add_feature(land_10m)

    # cb = plt.colorbar(dens, cax=pos_cax)
    # cb.set_label('Mean time spent in cell (days)', size=12)
    # a0.set_aspect('auto', adjustable=None)
    # a0.margins(x=-0.01, y=-0.01)
    # plt.savefig(fh['fig'][1], dpi=300)

    # # Anomalies
    # for i in range(4):
    #     f1, a1 = plt.subplots(1, 1, figsize=(20, 10),
    #                           subplot_kw={'projection': ccrs.PlateCarree()})
    #     f1.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)

    #     axpos = a1.get_position()
    #     pos_x = axpos.x0+axpos.width + 0.41
    #     pos_y = axpos.y0
    #     cax_width = 0.015
    #     cax_height = axpos.height

    #     pos_cax = f1.add_axes([pos_x, pos_y, cax_width, cax_height])

    #     # Set up the colormap (from https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar)
    #     cmap = cmr.fusion
    #     dens = a1.pcolormesh(lon_bnd, lat_bnd, seasonal_means[i]/annual_mean, cmap=cmap,
    #                          norm=colors.LogNorm(vmin=1/5, vmax=5))

    #     # Add cartographic features
    #     gl = a1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                       linewidth=0.5, color='black', linestyle='-', zorder=11)
    #     gl.xlabels_top = False
    #     gl.ylabels_right = False

    #     land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
    #                                             edgecolor='face',
    #                                             facecolor='black')
    #     a1.add_feature(land_10m)

    #     cb = plt.colorbar(dens, cax=pos_cax)
    #     season = ['winter', 'spring', 'summer', 'autumn']
    #     cb.set_label('Mean time spent in cell (days) ratio between ' + season[i] + ' and annual mean', size=12)
    #     a1.set_aspect('auto', adjustable=None)
    #     a1.margins(x=-0.01, y=-0.01)
    #     plt.savefig(fh['fig'][2] + '_' + season[i] + '.png' , dpi=300)
    #     plt.close()

    # print()
    # print('check')


    # Calculate losses from beaching and sinking
    loss_grid = calc_loss(lon_bnd, lat_bnd, mass_bnd, param, fh)

    print('Calculation complete')

    # Plotting
    f2, a2 = plt.subplots(1, 1, figsize=(20, 10),
                         subplot_kw={'projection': ccrs.PlateCarree()})

    f2.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    a2.set_ylim([-70, 70])

    # Set up the colorbar
    axpos = a2.get_position()
    pos_x = axpos.x0+axpos.width + 0.41
    pos_y = axpos.y0
    cax_width = 0.015
    cax_height = axpos.height

    pos_cax = f2.add_axes([pos_x, pos_y, cax_width, cax_height])

    # Plot the colormesh
    cmap = cmr.fall_r
    hist = a2.pcolormesh(lon_bnd, lat_bnd, loss_grid, cmap=cmap, vmin=0, vmax=1)

    # Add cartographic features
    gl = a2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='black', linestyle='-', zorder=11)
    gl.xlabels_top = False
    gl.ylabels_right = False

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='black')
    a2.add_feature(land_10m)

    # a0.add_feature(cfeature.LAND, facecolor='black', zorder=10)

    cb = plt.colorbar(hist, cax=pos_cax)
    cb.set_label('Mass fraction reaching Seychelles', size=12)
    a2.set_aspect('auto', adjustable=None)
    a2.margins(x=-0.01, y=-0.01)
    plt.savefig(fh['fig'][0], dpi=300)







