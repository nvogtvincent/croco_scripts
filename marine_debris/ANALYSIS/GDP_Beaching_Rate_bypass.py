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
         'depth_thresh': -15}        # Depth threshold for drifter to count as 'beached'

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
      'fig':   dirs['fig'] + 'GDP_beaching_rate.png'}

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

coast_time = np.load(dirs['gdp'] + 'processed_coast_time.npy')     # Cumulative time spent at coast by trajectory (s)
prox_beach_arr = np.load(dirs['gdp'] + 'processed_prox_criterion.npy') # Proximity beaching criterion
bath_beach_arr = np.load(dirs['gdp'] + 'processed_bath_criterion.npy') # Depth beaching criterion
meta_beach_arr = np.load(dirs['gdp'] + 'processed_meta_criterion.npy') # Meta beaching criterion
kaan_beach_arr = np.load(dirs['gdp'] + 'processed_kaan_criterion.npy') # Kaandorp beaching criterion


###############################################################################
# DERIVE BEACHING RATE ########################################################
###############################################################################

# For each time step i, (0.25, 0.5, ...), calculate the proportion of drifters
# that were at sea at that time step (i.e. drifters where coast_time >= value)
# that beached at that time step, F. The proportion of unbeached drifters at time
# step i is therefore (1-F1)*(1-F2)*...*(1-Fi).

max_ct = 15
num_samples = max_ct+1
time_axis = np.linspace(0, max_ct, num=max_ct+1)

beach_arr = []
beach_arr.append(prox_beach_arr) # Class 1 (proximity)
beach_arr.append(bath_beach_arr) # Class 1 (bathymetry)
beach_arr.append(meta_beach_arr) # Class 2 (meta)
beach_arr.append(kaan_beach_arr) # Class 3 (Kaandorp)

title_arr = ['Proximity criteria (<30m depth)',
             'Bathymetry criterion (<500m from coastline)',
             'GDP Death code',
             'Kaandorp criterion (>0 +ve elevation 1km to N/E/S/W)']

beach_arr[0][beach_arr[0] > 0] = 1
beach_arr[1][beach_arr[1] > 0] = 1
beach_arr[2][beach_arr[2] > 0] = 1
beach_arr[3][beach_arr[3] > 0] = 1

# Create figure
f, ax = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True, sharex=True)
ax = ax.reshape((-1))
ax2 = []

for j in range(4):
    # Reset arrays
    f_unbeached = np.zeros_like(time_axis)
    f_unbeached[0] = 1  # By definition
    f_beaching_per_ts = np.zeros_like(time_axis)

    for i in range(1, num_samples):
        time_i = time_axis[i]
        time_i_ub = 2*time_axis[i] - time_axis[i-1]

        # Calculate number of drifters that were in the dataset at time i
        drifters_total_i = np.sum(coast_time > time_i)

        # Calculate the number of drifters that beached at time i
        drifters_beaching_i = np.sum(beach_arr[j]*(coast_time >= time_i)*(coast_time < time_i_ub))

        # Calculate fraction beached at time i
        f_beaching_i = drifters_beaching_i/drifters_total_i
        f_beaching_per_ts[i] = f_beaching_i

        # Calculate the fraction of drifters remaining unbeached
        f_unbeached[i] = f_unbeached[i-1]*(1-f_beaching_i)

    # Calculate beaching rate using mean
    l_c_mean = np.mean(f_beaching_per_ts[1:]) # Divide by 4 to convert to days

    # Calculate beaching rate using regression
    M = time_axis[:, np.newaxis]
    p, res, rnk, s = lstsq(M, np.log(f_unbeached))
    l_c_reg = -p

    t_model = np.linspace(0, max_ct, num=100)
    f_model_reg = np.exp(-l_c_reg*t_model)
    f_model_mean = np.exp(-l_c_mean*t_model)

    ax[j].scatter(time_axis, f_unbeached, marker='x', c='k')
    ax[j].plot(t_model, f_model_reg, c='r', linestyle='-', linewidth=0.5)
    ax[j].plot(t_model, f_model_mean, c='b', linestyle='-', linewidth=0.5)

    ax[j].set_ylim([0, 1])
    ax[j].set_xlim([0, max_ct])

    ax[j].set_title(title_arr[j])
    ax[j].text(max_ct*0.98, 0.95, 'Beaching rate (Regression) = 1/' + str(round((1/l_c_reg)[0], 1)) + ' days',
               horizontalalignment='right', color='r')
    ax[j].text(max_ct*0.98, 0.92, 'Beaching rate (Mean fraction lost) = 1/' + str(round((1/l_c_mean), 1)) + ' days',
               horizontalalignment='right', color='b')

    ax[j].spines['top'].set_visible(False)
    ax[j].spines['right'].set_visible(False)

    ax2.append(ax[j].twinx())
    ax2[j].bar(time_axis, f_beaching_per_ts, alpha=0.3, width=(time_axis[1]-time_axis[0]),
               color='g')
    ax2[j].set_ylim([0, 0.3])
    ax2[j].set_xlim([0, max_ct])
    ax2[j].spines['top'].set_visible(False)
    ax2[j].spines['left'].set_visible(False)

    ax[j].set_zorder(ax2[j].get_zorder()+1)
    ax[j].patch.set_visible(False)

    if j >= 2:
        ax[j].set_xlabel('Time spent by drifter 1/12 degrees of the coast')

    if j == 0 or j == 2:
        ax[j].set_ylabel('Fraction of dFAD unbeached')
        ax2[j].spines['right'].set_visible(False)
        ax2[j].get_yaxis().set_visible(False)
    else:
        ax2[j].set_ylabel('Fraction of drifters afloat that beach during time interval')
        ax[j].spines['left'].set_visible(False)
        ax[j].get_yaxis().set_visible(False)

plt.savefig(fh['fig'], dpi=300)

