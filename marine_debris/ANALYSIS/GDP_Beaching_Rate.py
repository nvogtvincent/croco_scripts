#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script extracts a beaching rate from GDP data
@author: Noam Vogt-Vincent

@drifter_source: https://www.aoml.noaa.gov/phod/gdp/interpolated/data/all.php
@coastline_source: https://www.soest.hawaii.edu/pwessel/gshhg/
@bathymetry_source: https://www.gebco.net/data_and_products/gridded_bathymetry_data/

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
import xarray as xr
from osgeo import gdal, osr
from glob import glob
from osgeo import gdal
from tqdm import tqdm
from netCDF4 import Dataset, num2date
from shapely.geometry import Point


# Methodology:
# 1. Determine whether a trajectory ever approaches within 1/12 of the coast
# 2. Calculate the cumulative time the drifter spends within 1/12 of the coast
# 3. Determine whether drifter has beached, using the following two criteria:
#    a. Last drifter location is within 500m of the GSHHG coast
#    b. Last drifter location is in <30m water depth (GEBCO2021)
# 4. Calculate F(beach) as a function of cumulative time within 1/12 of the coast

# PARAMETERS
param = {'vel_max': 60*100/(3600*6)} # Based on a drifter swinging in a 60m radius

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/',
        'gdp': os.path.dirname(os.path.realpath(__file__)) + '/../GDP_DATA/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'gdp': sorted(glob(dirs['gdp'] + 'bd*.dat')),
      'gebco': dirs['grid'] + 'LOC/gebco_2021/GEBCO_2021.nc',
      'coast_005deg': dirs['grid'] + 'LOC/coastal_mask/coastal_mask_005.gpkg',
      'coast_083deg': dirs['grid'] + 'LOC/coastal_mask/GSHHS_h_L1_buffer083_res01.tif',
      'fig':   dirs['fig'] + 'GDP_beaching_rate.png',}

###############################################################################
# LOAD DATA ###################################################################
###############################################################################

# Load coarse + rasterised 0.083deg coastal mask ##############################
coast83 = xr.open_rasterio(fh['coast_083deg'])
bnd83_dx = np.mean(np.gradient(coast83.x))
bnd83_dy = np.mean(np.gradient(coast83.y))
coast83_bnd_lon = np.concatenate((coast83.x, [coast83.x.values[-1]+bnd83_dx])) - 0.5*bnd83_dx
coast83_bnd_lat = np.concatenate((coast83.y, [coast83.y.values[-1]+bnd83_dy])) - 0.5*bnd83_dy

# Load 0.005deg (500m) coastal mask ###########################################
coast5_obj = gpd.read_file(fh['coast_005deg'])

# Load GEBCO bathymetry #######################################################
bath_data = xr.open_dataset(fh['gebco'])

###############################################################################
# ANALYSE GDP TRAJECTORIES ####################################################
###############################################################################

# Keep track of some basic statistics
stats = {'rejected_latitude': 0,  # Rejected due to latitude bounds
         'rejected_no_coast': 0,  # Rejected due to no coastal intercept
         'coast_no_beach': 0,     # Valid, no beaching event
         'coast_beach_prox': 0,   # Valid, beaching event via proximity criterion
         'coast_beach_depth': 0,  # Valid, beaching event via depth criterion
         'coast_beach_both': 0,}  # Valid, beaching event via both criteria

coast_time = []   # Cumulative time spent at coast by trajectory (s)
beach_status = [] # Beaching status: 0 = no beach
                  #                  1 = proximity beach
                  #                  2 = depth beach
                  #                  3 = proximity + depth beach


for gdp_fh in fh['gdp']:
    # Read file
    df = pd.read_csv(gdp_fh, sep='\s+', header=None,
                     usecols=[0, 4, 5, 9],
                     names=['ID', 'lat', 'lon', 'vel'],)

    # Change longitude from 0-360 -> -180-180
    lon_ = df['lon'].values
    lon_[lon_ > 180] = lon_[lon_ > 180]-360
    df['lon'] = lon_

    # Extract a list of drifter IDs
    drifter_id_list = np.unique(df['ID'])

    # Loop through drifters
    for drifter_id in tqdm(drifter_id_list, total=len(drifter_id_list)):
        df_drifter = df.loc[df['ID'] == drifter_id].copy()
        df_drifter = df_drifter.reset_index(0)

        # Reject if drifter goes further than 60deg from equator
        if np.any(np.abs(df_drifter['lat']) > 60):
            # Reject due to latitude criterion
            stats['rejected_latitude'] += 1
        else:
            # Assess coastal status of each location
            df_drifter['coast'] = coast83.interp(coords={'x': xr.DataArray(df_drifter['lon'].values, dims='z'),
                                                         'y': xr.DataArray(df_drifter['lat'].values, dims='z')},
                                                 method='nearest').values[0]

            # Only continue with trajectories that ever went near the coast
            if np.sum(df_drifter['coast']) > 0:
                # Assess whether drifter meets beaching criteria
                lon_end = df_drifter['lon'].iloc[-1]
                lat_end = df_drifter['lat'].iloc[-1]

                # Test proximity criterion
                end_pos = Point(lon_end, lat_end)
                proximity_beach = coast5_obj.geometry.intersects(end_pos)[0]

                # Test depth criterion
                end_depth = bath_data.interp(coords={'lon': lon_end,
                                                     'lat': lat_end},
                                             method='linear')['elevation'].values
                depth_beach = True if end_depth > -30 else False

                if proximity_beach or depth_beach:
                    # If trajectory has beached, try to determine when exactly it
                    # beached (assume drifter is swinging in a 60m radius)
                    #
                    # Start at the penultimate time index and work backwards until
                    # velocity exceeds threshold

                    tidx = len(df_drifter)-1
                    beach_found = False
                    while not beach_found:
                        tidx -= 1
                        if df_drifter['vel'].iloc[tidx] > param['vel_max']:
                            beach_found = True

                    # Now calculate cumulative time by coasts at this time (in days)
                    drifter_time_at_coast = np.cumsum(df_drifter['coast'])[tidx+1]*0.25
                else:
                    drifter_time_at_coast = np.sum(df_drifter['coast'])*0.25

                # Write data
                if proximity_beach and depth_beach:
                    stats['coast_beach_both'] += 1
                    beach_status.append(3)
                elif proximity_beach:
                    stats['coast_beach_prox'] += 1
                    beach_status.append(1)
                elif depth_beach:
                    stats['coast_beach_depth'] += 1
                    beach_status.append(2)
                else:
                    stats['coast_no_beach'] += 1
                    beach_status.append(0)

                coast_time.append(drifter_time_at_coast)

                # if depth_beach:
                #     f, a0 = plt.subplots(1, 1, figsize=(10, 10),
                #                          subplot_kw={'projection': ccrs.PlateCarree()})

                #     a0.scatter(df_drifter['lon'], df_drifter['lat'], s=1)
                #     a0.scatter(df_drifter['lon'].loc[df_drifter['coast'] == 1],
                #                df_drifter['lat'].loc[df_drifter['coast'] == 1],
                #                c=df_drifter['coast'].loc[df_drifter['coast'] == 1],
                #                vmin=0, vmax=1)
                #     a0.coastlines()
                #     a0.set_xlim([np.min(df_drifter['lon']-1), np.max(df_drifter['lon'])+1])
                #     a0.set_ylim([np.min(df_drifter['lat']-1), np.max(df_drifter['lat']+1)])
                #     plt.show()
                #     print()

            else:
                # Reject due to no coastal intercept
                stats['rejected_no_coast'] += 1

###############################################################################
# DERIVE BEACHING RATE ########################################################
###############################################################################

beach_status = np.array(beach_status)
beach_status[beach_status > 0] = 1

coast_time = np.array(coast_time)
coast_time_beach = coast_time*beach_status

time_array_bnd = np.linspace(0, 20, num=11)
time_array =  0.5*(time_array_bnd[1:] + time_array_bnd[:-1]) # x axis for plot

# Now bin
n_drifters_per_ct_bin = np.histogram(coast_time, bins=time_array_bnd)[0]
n_unbeached_drifters_per_ct_bin = np.histogram(coast_time, bins=time_array_bnd, weights=1-beach_status)[0]

# Calculate unbeached fraction per bin
f_unbeached = np.zeros_like(time_array)
sig_threshold = 5 # Minimum number of drifters per bin for significance
for i in range(len(time_array)):
    if n_drifters_per_ct_bin[i] <= sig_threshold:
        f_unbeached[i] = np.nan
    else:
        f_unbeached[i] = n_unbeached_drifters_per_ct_bin[i]/n_drifters_per_ct_bin[i]

plt.scatter(time_array, f_unbeached)




