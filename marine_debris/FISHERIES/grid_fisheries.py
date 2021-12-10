#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script produces a gridded estimate of plastic inputs from fisheries from
GFW fishing effort data
(https://globalfishingwatch.org/data-download/datasets/public-fishing-effort)

@author: Noam Vogt-Vincent (2021)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
import pandas as pd
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from glob import glob
import time

###############################################################################
# Parameters ##################################################################
###############################################################################

# PARAMETERS
param = {'grid_res': 2.0,                          # Grid resolution in degrees
         'lon_range': [20, 100],                   # Longitude range for output
         'lat_range': [-60, 30],                   # Latitude range for output

         'dict_name': 'mmsi_dict.pkl',
         'out_name': 'fisheries_waste_flux.png'}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'data_obs': os.path.dirname(os.path.realpath(__file__)) + '/DATA/OBS/',
        'data_ref': os.path.dirname(os.path.realpath(__file__)) + '/DATA/REF/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/FIGURES/'}

# FILE HANDLES
fh = {'obs': [],
      'ref_raw': dirs['data_ref'] + 'fishing-vessels-v2.csv',
      'ref_proc': dirs['data_ref'] + 'mmsi_lookup.pkl',
      'fig': dirs['fig'] + param['out_name']}

for path, subdirs, files in os.walk(dirs['data_obs']):
    for name in files:
        fh['obs'].append(os.path.join(path, name))

# GEAR INTENSITY & DISSIPATION RATE
gear_intensity = {0: {'LOA': 153.0, # Purse seines
                      'GT': 30.0,
                      'kW': 16.5},
                  1: {'LOA': 12.8, # Trawlers
                      'GT': 8.3,
                      'kW': 0.45},
                  2: {'LOA': 33.8}} # Longlines

###############################################################################
# Functions ###################################################################
###############################################################################

# Generate MMSI dictionary
# The basic assumptions here are:
# (1) Within a vessel class, the mass of equipment carried is proportional to
#     some dimension of that vessel (i.e. gear_intensity * gear_dimension)
# (2) We are NOT comparing different vessel classes, so the gear_intensity must
#     only be constant within a vessel class
# (3) Within a vessel class, ships lose a constant proportion of their equipment
#     mass per year, so the loss per fishing hour is the mass carried divided
#     by the hours fished that year
# (4) Where estimates of the actual gear intensity based on multiple gear
#     dimensions exist, we can calculate gear equipment mass from different
#     gear dimensions (if data is missing)
# (5) Where estimates of the gear intensity for multiple/any gear dimensions
#     does not exist, gear equipment mass can only be calculated from one
#     particular gear dimension.
# (6) Where that particular gear dimension does not exist, we disregard the
#     vessel (class <- 'none')
# (7) We group vessels into the top-level classification by GFW:
#     https://globalfishingwatch.org/datasets-and-code-vessel-identity/
# (8) We assume that the gear carried by trawlers and dredgers is similar
#     enough to use the same gear intensity


def dictgen(file_in, file_out, gear_intensity):
    # Load file
    data = pd.read_csv(file_in,
                       usecols=['mmsi',
                                'vessel_class_gfw',
                                'length_m_gfw',
                                'engine_power_kw_gfw',
                                'tonnage_gt_gfw',
                                'fishing_hours_2012',
                                'fishing_hours_2013',
                                'fishing_hours_2014',
                                'fishing_hours_2015',
                                'fishing_hours_2016',
                                'fishing_hours_2017',
                                'fishing_hours_2018',
                                'fishing_hours_2019',
                                'fishing_hours_2020',])

    data = data.rename(columns={'vessel_class_gfw': 'class',
                                'length_m_gfw': 'LOA',
                                'engine_power_kw_gfw': 'kW',
                                'tonnage_gt_gfw': 'GT',
                                'fishing_hours_2012': '2012',
                                'fishing_hours_2013': '2013',
                                'fishing_hours_2014': '2014',
                                'fishing_hours_2015': '2015',
                                'fishing_hours_2016': '2016',
                                'fishing_hours_2017': '2017',
                                'fishing_hours_2018': '2018',
                                'fishing_hours_2019': '2019',
                                'fishing_hours_2020': '2020',})


    def calculate_lost_gear_mass_per_hr(row, gear_intensity):
        # Useful null list
        null = [0., 0., 0., 0., 0., 0., 0., 0., 0., 'none']

        # Firstly establish whether vessel is applicable (return 0 if not)
        # if pd.isnull(row['time']) or row['time'] == 0:
        #     return null # lpy = mass loss per year

        # Establish the vessel type, then estimate the gear carried
        vessel_class = row['class']

        # Function to convert mass loss per year to array of mass loss per hour
        def lpy2lph(lpy, row, vessel_type):
            # Quick lambda that calculates loss per hour from loss per year and
            # hours per year, and adds vessel type
            conv = lambda lpy, hpy : 0. if (pd.isnull(hpy) + (hpy == 0)) else lpy/hpy

            return [conv(lpy, row['2012']),
                    conv(lpy, row['2013']),
                    conv(lpy, row['2014']),
                    conv(lpy, row['2015']),
                    conv(lpy, row['2016']),
                    conv(lpy, row['2017']),
                    conv(lpy, row['2018']),
                    conv(lpy, row['2019']),
                    conv(lpy, row['2020']),
                    vessel_type]

        if "seine" in vessel_class:
            if not pd.isnull(row['LOA']):
                lpy = row['LOA']*gear_intensity[0]['LOA']
            # elif not pd.isnull(row['GT']):
            #     lpy = row['GT']*gear_intensity[0]['GT']
            # elif not pd.isnull(row['kW']):
            #     lpy = row['kW']*gear_intensity[0]['kW']
            else:
                return null

            return lpy2lph(lpy, row, 'seiners')

        elif "trawl" in vessel_class or "dredge" in vessel_class:
            if not pd.isnull(row['LOA']):
                lpy = row['LOA']*gear_intensity[1]['LOA']
            # elif not pd.isnull(row['GT']):
            #     lpy = row['GT']*gear_intensity[1]['GT']
            # elif not pd.isnull(row['kW']):
            #     lpy = row['kW']*gear_intensity[1]['kW']
            else:
                return null

            return lpy2lph(lpy, row, 'trawlers_and_dredgers')

        elif "set_longline" in vessel_class or "set_gillnet" in vessel_class or "pot" in vessel_class or "fixed" in vessel_class:
            if not pd.isnull(row['LOA']):
                lpy = row['LOA']*1
            else:
                return null

            return lpy2lph(lpy, row, 'fixed_gear')

        elif "drifting_longline" in vessel_class:
            if not pd.isnull(row['LOA']):
                lpy = row['LOA']*1
            else:
                return null

            return lpy2lph(lpy, row, 'drifting_longlines')

        elif "jigg" in vessel_class:
            if not pd.isnull(row['LOA']):
                lpy = row['LOA']*1
            else:
                return null

            return lpy2lph(lpy, row, 'squid_jigger')

        elif "pole" in vessel_class or "troll" in vessel_class:
            if not pd.isnull(row['LOA']):
                lpy = row['LOA']*1
            else:
                return null

            return lpy2lph(lpy, row, 'pole_and_line_and_trollers')

        else:
            if not pd.isnull(row['LOA']):
                lpy = row['LOA']*1
            else:
                return null

            return lpy2lph(lpy, row, 'fishing')

    # Calculate mass of gear carried by vessel
    data[['2012',
          '2013',
          '2014',
          '2015',
          '2016',
          '2017',
          '2018',
          '2019',
          '2020',
          'vessel_class']] = data.apply(lambda row: calculate_lost_gear_mass_per_hr(row, gear_intensity), axis=1, result_type='expand')

    data = data[['mmsi', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', 'vessel_class']]

    # Statistics
    print('Number of seiners: ' + str(len(data[data['vessel_class'] == 'seiners'])))
    print('Number of trawlers/dredgers: ' + str(len(data[data['vessel_class'] == 'trawlers_and_dredgers'])))
    print('Number of fixed_gear: ' + str(len(data[data['vessel_class'] == 'fixed_gear'])))
    print('Number of drifting_longlines: ' + str(len(data[data['vessel_class'] == 'drifting_longlines'])))
    print('Number of squid_jiggers: ' + str(len(data[data['vessel_class'] == 'squid_jigger'])))
    print('Number of pole_and_line_and_trollers: ' + str(len(data[data['vessel_class'] == 'pole_and_line_and_trollers'])))
    print('Number of unclassified: ' + str(len(data[data['vessel_class'] == 'fishing'])))
    print('Number of no_dimension: ' + str(len(data[data['vessel_class'] == 'none'])))

    data.to_pickle(file_out)

    return data

# Grid effort
# For each datapoint in the GFW dataset, we carry out the following steps:
# (1) Establish if datapoint is within domain, and if so, where
# (2) Find MMSI in dictionary and find mass moment per hour and vessel class
# (3) Grid datapoint


def grid_effort(file, grid, mmsi_lookup, lon_bnd, lat_bnd):
    # Get year
    year = file.split('/')[-1].split('-')[0]

    # Load data
    data = pd.read_csv(file,
                       usecols=['date',
                                'cell_ll_lat',
                                'cell_ll_lon',
                                'mmsi',
                                'fishing_hours'])

    # Filter out observations not in region, and observations with no fishing
    # hours
    data = data[(data['cell_ll_lat'] < 30) &
                (data['cell_ll_lat'] > -60) &
                (data['cell_ll_lon'] < 100) &
                (data['cell_ll_lon'] > 20) &
                (data['fishing_hours'] > 0)]

    data = data.reset_index(drop=True)

    # Filter the mmsi lookup dict to reduce runtime

    n_obs = len(data)

    # Match mmsi with vessel type and mass moment
    # For convenience, replace mmsi with vessel type and fishing_hours with
    # mass moment

    for i in range(n_obs):
        # Load row
        mmsi = data.iloc[i]['mmsi']
        hrs = data.iloc[i]['fishing_hours']

        # Find entry in dict
        mmsi_entry = mmsi_lookup[mmsi_lookup['mmsi'] == mmsi]

        # Write to data
        data.loc[i, 'mmsi'] = mmsi_entry['vessel_class'].values[0]
        data.loc[i, 'fishing_hours'] = mmsi_entry[year].values[0]*hrs

    # Now grid
    for i, vtype in enumerate(['seiners', 'trawlers_and_dredgers', 'fixed_gear',
                               'drifting_longlines', 'squid_jigger',
                               'pole_and_line_and_trollers', 'fishing', 'all']):

        if vtype == 'all':
            print()
        else:
            subdata = data[data['mmsi'] == vtype]
            # if len(subdata):
            grid[i, :, :] += np.histogram2d(subdata['cell_ll_lon'],
                                            subdata['cell_ll_lat'],
                                            bins=[lon_bnd, lat_bnd],
                                            weights=subdata['fishing_hours'])[0].T

    return grid


###############################################################################
# Run script ##################################################################
###############################################################################

if __name__ == '__main__':

    # Generate grid
    lon_bnd = np.linspace(param['lon_range'][0],
                          param['lon_range'][1],
                          int((param['lon_range'][1] -
                               param['lon_range'][0])/param['grid_res'])+1)

    lat_bnd = np.linspace(param['lat_range'][0],
                          param['lat_range'][1],
                          int((param['lat_range'][1] -
                               param['lat_range'][0])/param['grid_res'])+1)

    lon = 0.5*(lon_bnd[1:] + lon_bnd[:-1])
    lat = 0.5*(lat_bnd[1:] + lat_bnd[:-1])

    grid = np.zeros((8, len(lat), len(lon)), dtype=np.float64)

    # Generate MMSI dict
    try:
        mmsi_lookup = pd.read_pickle(fh['ref_proc'])
    except:
        mmsi_lookup = dictgen(fh['ref_raw'], fh['ref_proc'], gear_intensity)

    # Grid fishing effort
    for k, file in enumerate(fh['obs']):
        grid = grid_effort(file, grid, mmsi_lookup, lon_bnd, lat_bnd)
        print('{:.2f}'.format(100*k/len(fh['obs'])) + '%')


    print('Finished!')