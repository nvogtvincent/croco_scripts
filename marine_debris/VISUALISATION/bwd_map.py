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

         'out_name': 'inverse_map.png'}            # Output figure name

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)) + '/../',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/../TRAJ/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'traj': sorted(glob(dirs['traj'] + '*.nc')),
      'fig': dirs['fig'] + param['out_name']}

cheat = 100

###############################################################################
# Routines ####################################################################
###############################################################################


def get_traj(fh, fh_num, start, end):
    with Dataset(fh['traj'][fh_num], mode='r') as nc:
        lon_out = nc.variables['lon'][start:end, :]
        lat_out = nc.variables['lat'][start:end, :]
        ts_out = nc.variables['time'][start:end, :]
        tb_out = nc.variables['ct'][start:end, :]

        # Convert ts to time at sea
        ts_out = ts_out[:, 0].reshape(-1, 1) - ts_out

    return [lon_out.compressed(),
            lat_out.compressed(),
            ts_out.compressed(),
            tb_out.compressed()]


def grid_traj(lon_bnd, lat_bnd, param, fh):
    # Calculate available memory
    avail_mem = virtual_memory()[4]

    # Generate the empty grids
    tot_grid = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1), dtype=np.float32)
    num_grid = np.zeros((len(lat_bnd)-1, len(lon_bnd)-1), dtype=np.int32)

    # Loop over trajectory files
    for fh_num, traj_fh in enumerate(fh['traj']):
        # Calculate chunking for this trajectory file
        with Dataset(fh['traj'][0], mode='r') as nc:
            ntraj, ntime = np.shape(nc.variables['lon'])
            ntraj_ch = int(avail_mem/(ntime*8*cheat))
            n_ch = int(np.ceil(ntraj/ntraj_ch))

        # Loop over chunks
        for chunk in range(n_ch): # Change to n_ch
            # Load data
            data = get_traj(fh, fh_num,
                            ntraj_ch*chunk,
                            ntraj_ch*(chunk+1))

            # Calculate mass ratio at each point
            mass = np.exp((-param['lb']*data[3])+(-param['ls']*data[2]))
            tot_grid += np.histogram2d(data[1], data[0],
                                       (lat_bnd, lon_bnd), weights=mass)[0]
            num_grid += np.histogram2d(data[1], data[0],
                                       (lat_bnd, lon_bnd))[0].astype(np.int32)

    # Finally, calculate the average
    num_grid = np.ma.masked_where(num_grid == 0, num_grid)
    grid = tot_grid/num_grid

    return grid
















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

    print('Gridding trajectories...')
    inv_grid = grid_traj(lon_bnd, lat_bnd, param, fh)
    print('Calculation complete')
    plt.figure()
    plt.pcolormesh(inv_grid, norm=colors.LogNorm(vmin=0.001, vmax=1))
    plt.colorbar()



























##############################################################################
# Binning ####################################################################
##############################################################################

# def binModel(lon_bnd, lat_bnd, param, fh):
#     @njit
#     def update_min_time_model(lon_full, lat_full, time_full, grid, data):
#         lat_bnd = grid[0]
#         lon_bnd = grid[1]

#         ntraj = lon_full.shape[0]
#         for i in range(ntraj):
#             # Load positions and times for this trajectory
#             lon = lon_full[i, :]
#             lat = lat_full[i, :]
#             time = time_full[i, :]

#             mask = ~np.isnan(lon)
#             lon = lon[mask]
#             lat = lat[mask]
#             time = time[mask]

#             # Find indices on grid for all particle positions
#             lat_idx = np.searchsorted(lat_bnd, lat)-1
#             lon_idx = np.searchsorted(lon_bnd, lon)-1

#             for j in range(len(lon)):
#                 # Check if time in current grid bin is less than the previous min
#                 # If so, overwrite
#                 old_val = data[lat_idx[j], lon_idx[j]]
#                 if time[j] < old_val:
#                     data[lat_idx[j], lon_idx[j]] = time[j]

#         return data

#     # Generate binning grid
#     min_days = np.ones((len(lat_bnd)-1, len(lon_bnd)-1),
#                        dtype=np.float32)*32768

#     # Find the dimensions of the trajectory data
#     param['nParticleFile'] = len(fh['traj'])

#     for pf in range(param['nParticleFile']):
#         with Dataset(fh['traj'][pf], mode='r') as nc:
#             # Firstly calculate data chunking
#             ntime = nc.variables['trajectory'].shape[1]
#             ntraj = nc.variables['trajectory'].shape[0]

#             avail_mem = psutil.virtual_memory()[4]
#             c_ntraj   = np.ceil(avail_mem/(ntime*8*100)).astype('int') # Trajectories per chunk
#             c_n       = np.ceil(ntraj/c_ntraj).astype('int')          # Number of chunks

#             for i in range(c_n): #c_n
#                 # Load in chunk data
#                 c_lon  = nc.variables['lon'][i*c_ntraj:(i+1)*c_ntraj]
#                 c_lat  = nc.variables['lat'][i*c_ntraj:(i+1)*c_ntraj]
#                 c_time = nc.variables['time'][i*c_ntraj:(i+1)*c_ntraj]

#                 c_time = c_time - np.repeat(c_time[:, 0], c_time.shape[1]).reshape((-1, c_time.shape[1]))
#                 c_time = np.abs(c_time)/86400. # Convert to days at sea

#                 # Now update min_days
#                 # start = pytime.time()

#                 min_days = update_min_time_model(np.array(c_lon),
#                                                  np.array(c_lat),
#                                                  np.array(c_time).astype(np.int16),
#                                                  (lat_bnd, lon_bnd),
#                                                  min_days)
#                 # print(pytime.time()-start)

#     # min_days = np.ma.masked_equal(min_days, 32767)/365
#     min_days[min_days < 32768] = min_days[min_days < 32768]/365

#     return min_days

# def binGDP(lon_bnd, lat_bnd, param, fh):
#     @njit
#     def update_min_time_gdp(lon, lat, time, grid, data):
#         lat_bnd = grid[0]
#         lon_bnd = grid[1]

#         # Switch longitude from 0-360 to -180-180
#         lon[lon > 180] = lon[lon > 180]-360

#         # Remove NaNs
#         time = time[lon < 180]
#         lat = lat[lon < 180]
#         lon = lon[lon < 180]

#         lat_idx = np.searchsorted(lat_bnd, lat)-1
#         lon_idx = np.searchsorted(lon_bnd, lon)-1

#         for j in range(len(lon)):
#             # Check if time in current grid bin is less than the previous min
#             # If so, overwrite
#             try:
#                 old_val = data[lat_idx[j], lon_idx[j]]
#             except:
#                 print('fuck')
#             if time[j] < old_val:
#                 data[lat_idx[j], lon_idx[j]] = time[j]

#         return data

#     # Generate binning grid
#     min_days = np.ones((len(lat_bnd)-1, len(lon_bnd)-1),
#                        dtype=np.float32)*32767

#     for fhi_num, fhi in enumerate(fh['gdp']):
#         df = pd.read_csv(fhi, sep='\s+', header=None,
#                          usecols=[i for i in range(6)],
#                          names=['ID', 'month', 'day', 'year', 'lat', 'lon'],)
#                          ##skiprows=3950000, nrows=50000)


#         # Sort by geography
#         # Convert to numpy array first, then evaluate whether any coordinate
#         # pair is within region of interest
#         df = df.groupby('ID').agg(lambda x: list(x))[['lat', 'lon', 'year', 'month', 'day']]
#         df['ID'] = df.index
#         arr = np.array(df)
#         valid_id = []

#         for row in range(arr.shape[0]):
#             # CONVERT THIS TO A LAMBDA!
#             lat_valid = any((lat <= param['gdp_range1']['lat_max'])*
#                             (lat >= param['gdp_range1']['lat_min'])*
#                             (lon >= param['gdp_range1']['lon_min'])*
#                             (lon <= param['gdp_range1']['lon_max']) for lat, lon in zip(arr[row, 0], arr[row, 1]))

#             if lat_valid:
#                 valid_id.append(arr[row, -1])

#         valid_mask = np.in1d(arr[:, -1], valid_id)
#         arr = arr[valid_mask, :]

#         # Now find the last time that the trajectory was within the selected
#         # region, and cut
#         for row in range(arr.shape[0]):
#             in_sey = [(lat <= param['gdp_range1']['lat_max'])*
#                       (lat >= param['gdp_range1']['lat_min'])*
#                       (lon >= param['gdp_range1']['lon_min'])*
#                       (lon <= param['gdp_range1']['lon_max']) for lat, lon in zip(arr[row, 0], arr[row, 1])]

#             for var_idx in range(arr.shape[1]-1): # Exclude last column (index)
#                 arr[row, var_idx] = arr[row, var_idx][:np.max(np.nonzero(in_sey))]

#             # Subtract end time (sorry if anyone reads this method, it is very ugly)
#             time_arr = pd.DataFrame(np.array([np.array(i) for i in arr[row, 2:5]]).T,
#                                     columns=['year', 'month', 'day'])
#             time_arr['time'] = pd.to_datetime(time_arr)
#             time_arr['time'] = time_arr['time'] - time_arr['time'].iloc[-1]
#             time_arr['time'] = -time_arr['time'].dt.days

#             # Dump time_arr into arr
#             arr[row, 2] = np.array(time_arr['time'])

#         # Now calculate the time away from arriving at Seychelles
#         if fhi_num == 0:
#             sey_traj_ = np.copy(np.delete(arr, [3, 4], axis=1))
#         else:
#             sey_traj_ = np.concatenate((sey_traj_, np.delete(arr, [3, 4], axis=1)))

#     # Collapse and flatten arrays
#     sey_traj = []
#     for i in range(sey_traj_.shape[1]-1):
#         sey_traj.append(np.array([j for k in sey_traj_[:, i] for j in k]))


#     min_days = update_min_time_gdp(sey_traj[1],
#                                    sey_traj[0],
#                                    sey_traj[2],
#                                    (lat_bnd, lon_bnd),
#                                    min_days)

#     min_days[min_days < 32768] = min_days[min_days < 32768]/365

#     return min_days


# ##############################################################################
# # Plot #######################################################################
# ##############################################################################

# def plotModel(data1, data2, fh):
#     data = [data1, data2]

#     for i in range(2):
#         # First plot (model)
#         f, a0 = plt.subplots(1, 1, figsize=(20, 10),
#                              subplot_kw={'projection': ccrs.PlateCarree()})
#         f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)

#         # Set up the colorbar
#         axpos = a0.get_position()
#         pos_x = axpos.x0+axpos.width + 0.22
#         pos_y = axpos.y0
#         cax_width = 0.02
#         cax_height = axpos.height

#         pos_cax = f.add_axes([pos_x, pos_y, cax_width, cax_height])

#         # Set up the colormap (from https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar)
#         cmap = cmo.thermal_r
#         cmaplist = [cmap(i) for i in range(cmap.N)]
#         cmap = colors.LinearSegmentedColormap.from_list('dense_sq', cmaplist, cmap.N)
#         cmap_bounds = np.linspace(0, 10, 11)
#         cmap_norm = colors.BoundaryNorm(cmap_bounds, cmap.N)

#         # Plot the colormesh
#         hist = a0.pcolormesh(lon_bnd, lat_bnd, data[i], cmap=cmap, norm=cmap_norm,
#                              vmin=0, vmax=10)

#         # Add cartographic features
#         gl = a0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                           linewidth=0.5, color='black', linestyle='-', zorder=11)
#         gl.xlabels_top = False
#         gl.ylabels_right = False

#         a0.add_feature(cfeature.LAND, facecolor='white', zorder=10)

#         cb = plt.colorbar(hist, cax=pos_cax)
#         cb.set_label('Minimum years at sea to Seychelles', size=12)
#         a0.set_aspect('auto', adjustable=None)
#         a0.margins(x=-0.01, y=-0.01)
#         plt.savefig(fh['fig'][i], dpi=300)

#     # Second plot (observations)

# if __name__ == '__main__':
#     # Generate a global grid
#     lon_bnd = np.linspace(param['lon_range'][0],
#                           param['lon_range'][1],
#                           int((param['lon_range'][1] - param['lon_range'][0])/param['res'])+1)

#     lat_bnd = np.linspace(param['lat_range'][0],
#                           param['lat_range'][1],
#                           int((param['lat_range'][1] - param['lat_range'][0])/param['res'])+1)

#     model_im = binModel(lon_bnd, lat_bnd, param, fh)
#     gdp_im = binGDP(lon_bnd, lat_bnd, param, fh)

#     plotModel(model_im, gdp_im, fh)