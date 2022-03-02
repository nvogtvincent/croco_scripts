#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script extracts a beaching rate from dFAD data
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
import pyreadr
import rpy2.robjects as robjects
from scipy.linalg import lstsq
from glob import glob
from tqdm import tqdm
from shapely.geometry import Point
from datetime import datetime
import cartopy.crs as ccrs

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
        'dfad': os.path.dirname(os.path.realpath(__file__)) + '/../GDP_DATA/dFADs/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'dfad': [dirs['dfad'] + 'dfad_trajs.pkl',
               dirs['dfad'] + 'dfad_beachings.pkl',
               dirs['dfad'] + 'dfad_deployments.pkl',
               dirs['dfad'] + 'dfad_pts.pkl'],
      'gebco': dirs['grid'] + 'LOC/gebco_2021/GEBCO_2021.nc',
      'coast_005deg': dirs['grid'] + 'LOC/coastal_mask/coastal_mask_005.gpkg',
      'coast_083deg': dirs['grid'] + 'LOC/coastal_mask/GSHHS_h_L1_buffer083_res005.tif',
      'fig':   dirs['fig'] + 'GDP_beaching_rate.png',
      'data_out': dirs['dfad'] + 'processed_'}

###############################################################################
# LOAD DATA ###################################################################
###############################################################################

# Load dFAD data ##############################################################
dfad_data_all = {'info': pd.read_pickle(fh['dfad'][0]),
                 'beach': pd.read_pickle(fh['dfad'][1]),
                 'deploy': pd.read_pickle(fh['dfad'][2]),
                 'traj': pd.read_pickle(fh['dfad'][3])}

dfad_data_all['info']['strt_dt'] = pd.to_datetime(dfad_data_all['info']['strt_dt'], format='%Y-%m-%d')
dfad_data_all['info']['end_dat'] = pd.to_datetime(dfad_data_all['info']['end_dat'], format='%Y-%m-%d')
dfad_data_all['beach']['frst_p_'] = pd.to_datetime(dfad_data_all['beach']['frst_p_'], format='%Y-%m-%d')
dfad_data_all['beach']['lst_pt_'] = pd.to_datetime(dfad_data_all['beach']['lst_pt_'], format='%Y-%m-%d')
dfad_data_all['deploy']['pt_date'] = pd.to_datetime(dfad_data_all['deploy']['pt_date'], format='%Y-%m-%d')
dfad_data_all['traj']['pt_date'] = pd.to_datetime(dfad_data_all['traj']['pt_date'], format='%Y-%m-%d')

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
# ANALYSE dFAD TRAJECTORIES ###################################################
###############################################################################

# Create a list of buoy IDs
id_list = np.unique(dfad_data_all['traj']['buoy_id'])
id_list_beached = np.unique(dfad_data_all['beach']['buoy_id'])
id_list_beached = id_list_beached[np.isin(id_list_beached, id_list)]

# Keep track of some basic statistics
stats = {'rejected_no_coast': 0, # Rejected due to no coastal intercept
         'coast': 0,}            # Valid, no beaching event

coast_time_arr = []              # Cumulative time spent at coast by trajectory for terminal beaching events only (s)
# coast_time_Iall_arr = []       # Cumulative time spent at coast by trajectory for all beaching events via Imzilen criterion (s)

imzilen_beach_arr = []           # Whether trajectory beached or not according to Imzilen 2021 criterion, termination only
# imzilen_all_beach_arr = []     # Whether trajectory beached or not according to Imzilen 2021 criterion, all beaching events
prox_beach_arr = []              # Proximity beaching criterion
bath_beach_arr = []              # Depth beaching criterion
kaandorp_beach_arr = []          # Quasi-depth criterion similar to Kaandorp 2020, i.e. check if points 30 arc-seconds to the N/E/S/W of last location has a positive elevation

for i, dfad_id in enumerate(id_list):
    print(dfad_id)
    dfad_data = {'info': dfad_data_all['info'].loc[dfad_data_all['info']['buoy_id'] == dfad_id],
                 'beach': dfad_data_all['beach'].loc[dfad_data_all['beach']['buoy_id'] == dfad_id],
                 'deploy': dfad_data_all['deploy'].loc[dfad_data_all['deploy']['buoy_id'] == dfad_id],
                 'traj': dfad_data_all['traj'].loc[dfad_data_all['traj']['buoy_id'] == dfad_id]}

    # Firstly calculate the number of deployments
    deployment_sections = np.unique(dfad_data['traj']['sctn_nm'])
    deployment_n = len(deployment_sections)

    for j, sec in enumerate(deployment_sections):
        pos = dfad_data['traj'].loc[dfad_data['traj']['sctn_nm'] == sec].copy()
        pos_beach = dfad_data['beach'].loc[(dfad_data['beach']['frst_p_'] >= pos['pt_date'].iloc[0]) &
                                           (dfad_data['beach']['lst_pt_'] <= pos['pt_date'].iloc[-1])]

        # Evaluate coastal status of trajectory locations
        coast_intersection = coast83.interp(coords={'x': xr.DataArray(pos['geometry'].x.values, dims='z'),
                                                    'y': xr.DataArray(pos['geometry'].y.values, dims='z')},
                                            method='nearest').values[0]

        if np.sum(coast_intersection) > 0:
            # Estimate the time represented by each time frame
            # Firstly test if this is a regularly broadcasting dFAD (return [COUNTS PER DAY] [OCCURANCES])
            freq_counts = np.unique(np.unique(pos['pt_date'], return_counts=True)[1], return_counts=True)

            # If > 95% of occurances are a constant count per day, then assume this is a regularly broadcasting dFAD
            # However, only do this if that frequency is the highest frequency available (otherwise you could)
            # end up with >24 hours being represented by one day
            if np.max(freq_counts[1])/np.sum(freq_counts[1]) > 0.95 and freq_counts[0][np.argmax(freq_counts[1])] == np.max(freq_counts[0]):
                # Find the constant frequency
                const_freq = freq_counts[0][np.argmax(freq_counts[1])]
                const_freq = 24/const_freq

                # Convert coast_intersection from bool to time (hrs)
                coast_int_time = coast_intersection*const_freq
            else:
                # Otherwise, calculate the time per event as the hours per day divided by
                # the number of events for that day.
                coast_events = np.where(coast_intersection == 1)
                coast_int_time = np.zeros_like(coast_intersection)

                for k in coast_events:
                    k_time = pos['pt_date'].values[k]
                    k_occurances = pos['pt_date'].value_counts()[k_time]
                    coast_int_time[k] = 24/k_occurances

            pos['time_at_coast'] = coast_int_time

            # Now that we have calculated the time spent on the coast, establish the various
            # beaching criteria

            # METHODOLOGY
            # 1. Firstly, check whether there are any mid-trajectory beaching events. If
            #    so, remove them from the array.
            # 2. Secondly, check whether the end of the trajectory coincides with an Imzilen 21
            #    beaching event. If so, calculate the cumulative amount of time spent before
            #    the beaching event starts.
            # 3. Thirdly, assess the remaining criteria and, if they are met, backtrack
            #    until they are no longer met to find the start of that beaching event. Note that
            #    this was not needed for GDP data because those were already fully QAd.

            # Firstly, check whether a beaching event as defined by Imzilen 21 coincides with the
            # end of the trajectory

            if len(pos_beach):
                imzilen_beach_status = False
                traj_end = pos['pt_date'].iloc[-1]

                for row in range(pos_beach):
                    if pos_beach['frst_p_'].iloc[row] >= traj_end and pos_beach['lst_pt_'].iloc[row] >= traj_end:
                        imzilen_beach_status = True
                        pos2 = pos.loc[pos['pt_date'] <= pos_beach['frst_p_']]
                        break

                if imzilen_beach_status:
                    imzilen_beach_arr.append(pos2['time_at_coast'])
                else:




                print()












            # Firstly, check the (precomputed) criteria from Imzilen 2021
            beach_events_in_section = len(pos_beach)

            # for beach_event in range(beach_events_in_section):
            #     if beach_event == 0:
            #         # If first beach event, start of time-series is start of section
            #         section_start = pos['pt_date'].values[0]
            #     else:
            #         # Otherwise, take the day after the last beaching ended
            #         section_start = pos_beach['lst_pt_'].iloc[beach_event-1]
            #         section_start += np.timedelta64(1, 'D')

            #     section_end = pos_beach['frst_p_'].values[beach_event]

            #     if section_end > section_start:
            #         # This is required because there are some cases where a
            #         # beaching event is identified immediately after the preceding
            #         # beaching event

            #         # Extract this section
            #         pos2 = pos.loc[(pos['pt_date'] >= section_start) &
            #                        (pos['pt_date'] <= section_end)]

            #         # Add the number of hours the dFAD spent in the coastal environment
            #         # before beaching
            #         coast_time_Iall_arr.append(np.sum)






        else:
            # Firstly check that there are no recorded beaching events. If there are, this is suspicious (although not impossible).
            if len(pos_beach) > 1:
                print('WARNING!')
                print('Records indicate a beaching event occured for buoy ID ' + str(dfad_id) + ' section ' + str(sec))
                print('But no coastal intercept was found!')

            # Dismiss
            stats['rejected_no_coast'] += 1






            #
            # unique_times = np.unique(pos['pt_date'].values, return_counts=True)

            # # Check when beaching events occur
            # if dfad_id in id_list_beached:
            #     beaching_events = dfad_data['beach'].loc[(dfad_data['beach']['frst_p_'] >= unique_times[0][0]) &
            #                                              (dfad_data['beach']['lst_pt_'] <= unique_times[0][-1])]
            #     print(0)







            # Now evaluate the length of time spent in the vicinity of the coast




            # Firstly check if this is a simple
            # print(np.unique((dfad_data['traj']['pt_date'].iloc[1:].values-dfad_data['traj']['pt_date'].iloc[:-1].values)/1e9))




    # deployments = np.shape(dfad_data['deploy'])[0]

    # # Check that the time-step is the same between all steps
    # dt_check = (dfad_data['traj']['pt_date'].iloc[1:].values-dfad_data['traj']['pt_date'].iloc[:-1].values)/1e9
    # # assert np.sum(~np.isin(dt_check.astype('float'), [0, 86400])) == 0

    # if np.sum(~np.isin(dt_check.astype('float'), [0, 86400])) != 0:
    #     print('Non-constant time step.')

    # # Check that the deployment time agrees with the first position
    # if dfad_data['deploy']['pt_date'].iloc[0] != dfad_data['traj']['pt_date'].iloc[0]:
    #     print('Deployment time does not agree with first position.')

    # deployment_sections = np.unique(dfad_data['traj']['sctn_nm'])
    # for deployment in range(len(deployment_sections)):
    #     sec = deployment_sections[deployment]
    #     deployment_start = dfad_data['traj']['pt_date'].loc[dfad_data['traj']['sctn_nm'] == sec].iloc[0]
    #     assert np.isin(dfad_data['deploy']['pt_date'], deployment_start).any() # Check that start of deployment is a deployment event


    # print()




    # for j in deployments:
    #     dfad_pos = dfad_arr[3]

    #     # Evaluate time spent at coasts



    # # Loop through drifters
    # for drifter_id in tqdm(drifter_id_list, total=len(drifter_id_list)):
    #     df_drifter = df.loc[df['ID'] == drifter_id].copy()
    #     df_drifter = df_drifter.reset_index(0)

    #     # Reject if drifter goes further than 60deg from equator
    #     if np.any(np.abs(df_drifter['lat']) > 60):
    #         # Reject due to latitude criterion
    #         stats['rejected_latitude'] += 1
    #     else:
    #         # Assess coastal status of each location
    #         df_drifter['coast'] = coast83.interp(coords={'x': xr.DataArray(df_drifter['lon'].values, dims='z'),
    #                                                      'y': xr.DataArray(df_drifter['lat'].values, dims='z')},
    #                                              method='nearest').values[0]

    #         # Only continue with trajectories that ever went near the coast
    #         if np.sum(df_drifter['coast']) > 0:
    #             # Assess whether drifter meets beaching criteria
    #             lon_end = df_drifter['lon'].iloc[-1]
    #             lat_end = df_drifter['lat'].iloc[-1]

    #             # Test proximity criterion
    #             end_pos = Point(lon_end, lat_end)
    #             proximity_beach = coast5_obj.geometry.intersects(end_pos)[0]

    #             # Test depth criterion
    #             end_depth = bath_data.interp(coords={'lon': lon_end,
    #                                                  'lat': lat_end},
    #                                          method='linear')['elevation'].values
    #             depth_beach = True if end_depth > -30 else False

    #              # Test Kaandorp criterion
    #             neighbouring_points = np.zeros((4,))
    #             displacement=0.00833333 # 30 arc-seconds in degrees
    #             neighbouring_points[0] = bath_data.interp(coords={'lon': lon_end-displacement, 'lat': lat_end}, method='linear')['elevation'].values
    #             neighbouring_points[1] = bath_data.interp(coords={'lon': lon_end+displacement, 'lat': lat_end}, method='linear')['elevation'].values
    #             neighbouring_points[2] = bath_data.interp(coords={'lon': lon_end, 'lat': lat_end-displacement}, method='linear')['elevation'].values
    #             neighbouring_points[3] = bath_data.interp(coords={'lon': lon_end, 'lat': lat_end+displacement}, method='linear')['elevation'].values

    #             kaandorp_beach = True if np.max(neighbouring_points) > 0 else False

    #             # Test metadata criterion
    #             if len(beaching_p.loc[beaching_p['ID']==drifter_id]):
    #                 # Firstly test if in beaching likelihood file
    #                 meta_beach = beaching_p['beaching_p'].loc[beaching_p['ID']==drifter_id].values[0] > param['beaching_p_thresh']
    #             else:
    #                 # Otherwise just use death code
    #                 meta_beach = dfm['death_code'].loc[dfm['ID']==drifter_id].values[0] == 1

    #             # Calculate cumulative time at coast
    #             drifter_time_at_coast = np.sum(df_drifter['coast'])*0.25

    #             # Write data
    #             prox_beach_arr.append(1) if proximity_beach else prox_beach_arr.append(0)
    #             bath_beach_arr.append(1) if depth_beach else bath_beach_arr.append(0)
    #             meta_beach_arr.append(1) if meta_beach else meta_beach_arr.append(0)
    #             kaandorp_beach_arr.append(1) if kaandorp_beach else kaandorp_beach_arr.append(0)

    #             if proximity_beach + depth_beach + meta_beach + kaandorp_beach:
    #                 stats['coast_beach'] += 1
    #             else:
    #                 stats['coast_no_beach'] += 1

    #             coast_time.append(drifter_time_at_coast)

    #         else:
    #             # Reject due to no coastal intercept
    #             stats['rejected_no_coast'] += 1

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




