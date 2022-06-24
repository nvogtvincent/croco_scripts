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
import geopandas as gpd
import pandas as pd
import xarray as xr
from scipy.linalg import lstsq
from glob import glob
from tqdm import tqdm
from shapely.geometry import Point


# Methodology:
# 1. Determine whether a trajectory ever approaches within 1/12 of the coast
# 2. Calculate the cumulative time the drifter spends within 1/12 of the coast
# 3. Determine whether drifter has beached, using the following two criteria:
#    a. Last drifter location is within 500m of the GSHHG coast
#    b. Last drifter location is in <30m water depth (GEBCO2021)
# 4. Calculate F(beach) as a function of cumulative time within 1/12 of the coast

# PARAMETERS
param = {'beaching_p_thresh': 0.90,  # Probability threshold in meta file for drifter to count as 'beached'
         'depth_thresh': -30}        # Depth threshold for drifter to count as 'beached'

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/',
        'gdp': os.path.dirname(os.path.realpath(__file__)) + '/../GDP_DATA/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'gdp': sorted(glob(dirs['gdp'] + 'buoydata*.dat')),
      'gdp_meta': sorted(glob(dirs['gdp'] + 'meta/dirfl*.dat.txt')),
      'gdp_beaching_p': dirs['gdp'] + 'meta/beaching_p.dat',
      'gebco': dirs['grid'] + 'LOC/gebco_2021/GEBCO_2021.nc',
      'coast_005deg': dirs['grid'] + 'LOC/coastal_mask/coastal_mask_005.gpkg',
      'coast_083deg': dirs['grid'] + 'LOC/coastal_mask/GSHHS_h_L1_buffer083_res005.tif',
      'fig':   dirs['fig'] + 'GDP_beaching_rate.png',
      'data_out': dirs['gdp'] + 'processed_'}

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

# Load overarching metadata
beaching_p = pd.read_csv(fh['gdp_beaching_p'], sep='\s+', header=None,
                         usecols=[0, 7], names=['ID', 'beaching_p'])

###############################################################################
# ANALYSE GDP TRAJECTORIES ####################################################
###############################################################################

# Keep track of some basic statistics
stats = {'rejected_latitude': 0,  # Rejected due to latitude bounds
         'rejected_no_coast': 0,  # Rejected due to no coastal intercept
         'coast_no_beach': 0,     # Valid, no beaching event
         'coast_beach': 0}        # Valid, beaching event

coast_time = []     # Cumulative time spent at coast by trajectory (s)
prox_beach_arr = [] # Proximity beaching criterion
bath_beach_arr = [] # Depth beaching criterion
meta_beach_arr = [] # Meta beaching criterion
kaandorp_beach_arr = [] # Quasi-depth criterion similar to Kaandorp 2021, i.e. check if points 30 arc-seconds to the N/E/S/W of last location has a positive elevation

for gdp_fh, gdp_meta_fh in zip(fh['gdp'], fh['gdp_meta']):
    # Read files
    df = pd.read_csv(gdp_fh, sep='\s+', header=None,
                     usecols=[0, 4, 5, 9],
                     names=['ID', 'lat', 'lon', 'vel'],)

    dfm = pd.read_csv(gdp_meta_fh, sep='\s+', header=None,
                      usecols=[0, 14],
                      names=['ID', 'death_code'],)

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

                 # Test Kaandorp criterion
                neighbouring_points = np.zeros((4,))
                displacement=0.00833333 # 30 arc-seconds in degrees
                neighbouring_points[0] = bath_data.interp(coords={'lon': lon_end-displacement, 'lat': lat_end}, method='linear')['elevation'].values
                neighbouring_points[1] = bath_data.interp(coords={'lon': lon_end+displacement, 'lat': lat_end}, method='linear')['elevation'].values
                neighbouring_points[2] = bath_data.interp(coords={'lon': lon_end, 'lat': lat_end-displacement}, method='linear')['elevation'].values
                neighbouring_points[3] = bath_data.interp(coords={'lon': lon_end, 'lat': lat_end+displacement}, method='linear')['elevation'].values

                kaandorp_beach = True if np.max(neighbouring_points) > 0 else False

                # Test metadata criterion
                if len(beaching_p.loc[beaching_p['ID']==drifter_id]):
                    # Firstly test if in beaching likelihood file
                    meta_beach = beaching_p['beaching_p'].loc[beaching_p['ID']==drifter_id].values[0] > param['beaching_p_thresh']
                else:
                    # Otherwise just use death code
                    meta_beach = dfm['death_code'].loc[dfm['ID']==drifter_id].values[0] == 1

                # Calculate cumulative time at coast
                drifter_time_at_coast = np.sum(df_drifter['coast'])*0.25

                # Write data
                prox_beach_arr.append(1) if proximity_beach else prox_beach_arr.append(0)
                bath_beach_arr.append(1) if depth_beach else bath_beach_arr.append(0)
                meta_beach_arr.append(1) if meta_beach else meta_beach_arr.append(0)
                kaandorp_beach_arr.append(1) if kaandorp_beach else kaandorp_beach_arr.append(0)

                if proximity_beach + depth_beach + meta_beach + kaandorp_beach:
                    stats['coast_beach'] += 1
                else:
                    stats['coast_no_beach'] += 1

                coast_time.append(drifter_time_at_coast)

            else:
                # Reject due to no coastal intercept
                stats['rejected_no_coast'] += 1

np.save(fh['data_out'] + 'prox_criterion.npy', prox_beach_arr)
np.save(fh['data_out'] + 'bath_criterion.npy', bath_beach_arr)
np.save(fh['data_out'] + 'meta_criterion.npy', meta_beach_arr)
np.save(fh['data_out'] + 'kaan_criterion.npy', kaandorp_beach_arr)
np.save(fh['data_out'] + 'coast_time.npy', coast_time)

###############################################################################
# DERIVE BEACHING RATE ########################################################
###############################################################################

max_ct = 20
ct_int = 11

bath_beach_arr = np.array(bath_beach_arr)
prox_beach_arr = np.array(prox_beach_arr)
meta_beach_arr = np.array(meta_beach_arr)
coast_time = np.array(coast_time)

# Create three beaching classes: bath+prox, meta, bath+prox+metae
beach_arr = []
beach_arr.append(bath_beach_arr + prox_beach_arr) # Class 1 (manual)
beach_arr.append(meta_beach_arr) # Class 2 (meta)
beach_arr.append(bath_beach_arr + prox_beach_arr + meta_beach_arr) # Class 3 (all)

title_arr = ['Bathymetry and proximity criteria',
             'GDP Death Code',
             'All criteria']

beach_arr[0][beach_arr[0] > 0] = 1
beach_arr[1][beach_arr[1] > 0] = 1
beach_arr[2][beach_arr[2] > 0] = 1

# Create time array (i.e. x axis, limits for binning)
time_array_bnd = np.linspace(0, max_ct, num=ct_int)
time_array =  0.5*(time_array_bnd[1:] + time_array_bnd[:-1]) # x axis for plot

# Create figure
f, ax = plt.subplots(3, 1, figsize=(8, 16), constrained_layout=True, sharex=True)

for i in range(3):
    # Carry out calculations:
    # P(unbeached drifters for coastal time T) = N(unbeached drifters for coastal time T) /
    #                                            N(total drifters for coastal time T)

    # Calculate number of total drifters per CT bin by just binning drifter total
    # CT
    # Calculate number of unbeached drifters pr CT by binning drifter total CT but
    # multiplied by a 1 if they are unbeached and by a 0 if they are beached

    n_drifters_per_ct_bin = np.histogram(coast_time, bins=time_array_bnd)[0]
    n_unbeached_drifters_per_ct_bin = np.histogram(coast_time, bins=time_array_bnd, weights=1-beach_arr[i])[0]

    # Calculate P(unbeached drifters for coastal time T)
    f_unbeached = np.zeros_like(time_array)
    sig_threshold = 5 # Minimum number of drifters per bin for significance
    for j in range(len(time_array)):
        if n_drifters_per_ct_bin[j] <= sig_threshold:
            f_unbeached[j] = np.nan
        else:
            f_unbeached[j] = n_unbeached_drifters_per_ct_bin[j]/n_drifters_per_ct_bin[j]

    # Calculate beaching rate
    nan_loc = np.where(np.isnan(f_unbeached)) # Remove any NaNs for LSTSQ
    M = time_array[:, np.newaxis]
    p, res, rnk, s = lstsq(np.delete(M, nan_loc).reshape((-1,1)),
                           np.delete(np.log(f_unbeached).reshape((-1,1)), nan_loc))
    l_c = -p

    t_model = np.linspace(0, max_ct, num=100)
    f_model = np.exp(-l_c*t_model)

    ax[i].scatter(time_array, f_unbeached, marker='x', c='k')
    ax[i].plot(t_model, f_model, c='r', linestyle='-', linewidth=0.5)
    ax[i].set_ylabel('Fraction of drifters unbeached')

    ax[i].set_ylim([0, 1])
    ax[i].set_xlim([0, max_ct])

    ax[i].set_title(title_arr[i])
    ax[i].text(max_ct*0.98, 0.95, 'Beaching rate = 1/' + str(round((1/l_c)[0], 1)) + ' days',
               horizontalalignment='right')

    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

    if i == 2:
        ax[i].set_xlabel('Time spent by drifter 1/12 degrees of the coast')











# beach_status = np.array(beach_status)
# beach_status[beach_status > 0] = 1


# coast_time_beach = coast_time*beach_status



# # Now bin
# n_drifters_per_ct_bin = np.histogram(coast_time, bins=time_array_bnd)[0]
# n_unbeached_drifters_per_ct_bin = np.histogram(coast_time, bins=time_array_bnd, weights=1-beach_status)[0]

# # Calculate unbeached fraction per bin
# f_unbeached = np.zeros_like(time_array)
# sig_threshold = 5 # Minimum number of drifters per bin for significance
# for i in range(len(time_array)):
#     if n_drifters_per_ct_bin[i] <= sig_threshold:
#         f_unbeached[i] = np.nan
#     else:
#         f_unbeached[i] = n_unbeached_drifters_per_ct_bin[i]/n_drifters_per_ct_bin[i]

# # Now calculate lambda:
# # F(t) = exp(-lambda*t) -> lambda : slope of ln(F(t)) against t
# M = time_array[:, np.newaxis]
# p, res, rnk, s = lstsq(M, np.log(f_unbeached))
# l_c = -p

# t_model = np.linspace(0, 20, num=11)
# f_model = np.exp(-l_c*t_model)

# a0.scatter(time_array, f_unbeached, marker='.', c='k')
# a0.plot(t_model, f_model, c='r', linestyle='--')




