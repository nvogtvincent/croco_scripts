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
import matplotlib.colors as colors
import cmasher as cmr
import pandas as pd
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from glob import glob
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature

###############################################################################
# Parameters ##################################################################
###############################################################################

# PARAMETERS
param = {'grid_res': 1.0,                          # Grid resolution in degrees
         'lon_range': [20, 100],                   # Longitude range for output
         'lat_range': [-50, 30],                   # Latitude range for output

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

gear_intensity = {'seine': 1,
                  'trawl': 1,
                  'dredge': 1,
                  'set_longline': 1,
                  'set_gillnet': 1,
                  'pot': 1,
                  'fixed': 1,
                  'drifting_longline': 1,
                  'jigg': 1,
                  'pole': 1,
                  'troll': 1}

# 0: {'LOA': 153.0, # Purse seines
#     'GT': 30.0,
#     'kW': 16.5},
# 1: {'LOA': 12.8, # Trawlers
#     'GT': 8.3,
#     'kW': 0.45},
# 2: {'LOA': 33.8}} # Longlines

###############################################################################
# Functions ###################################################################
###############################################################################

# Generate MMSI dictionary
# The basic assumptions here are:
# (1) Within a vessel class, the mass of equipment carried is proportional to
#     some dimension of that vessel (i.e. gear_intensity * gear_dimension)
# (2) Intensity is constant within a vessel class
# (3) We group vessels into the top-level classification by GFW:
#     https://globalfishingwatch.org/datasets-and-code-vessel-identity/

# ASSUMPTION CLASS A ('constant loss per year')
# Assume that all ships lose the same proportion of mass per year, i.e. loss per
# hour = mass of equipment on ship * proportion lost per year / hours per year

# ASSUMPION CLASS B ('constant loss per hour')
# Assume that mass loss is proportional to fishing time, i.e. loss per hour =
# mass of equipment on ship * proportion lost per hour

# In both cases, we just ignore this mass loss constant as it can be built into
# the gear intensity if we have sensible values.

# Calculation:
# Input: intensity (kg/yr/m but normalised to 1), fishing hours in year (HY)
# Output: mass loss to ocean (kg/hr)

# Assumption class A
# Mass loss per year (kg/yr) = intensity (kg/yr/m) * LOA (m)
# Mass loss per hour (kg/hr) = mass loss per year (kg/yr) / hours per year (hr/yr)
# i.e. Mass loss per hour is propto intensity * LOA / hours per year

# Assumption class B
# Mass loss per hour (kg/hr) = intensity (kg/yr/m) * LOA (m) * constant (yr/hr)
# i.e. Mass loss per hour is propto intensity * LOA

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


    def calculate_lost_gear_mass_per_hr(row):
        # Useful null list
        null = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], 'none']

        # Establish the vessel type, then estimate the gear carried
        vessel_class = row['class']

        # Function to convert mass loss per year to array of mass loss per hour
        def mass_loss_rate(mass, row, vessel_type):
            # Quick lambda that calculates loss per hour under [assumption A, assumption B]
            conv = lambda mass, hpy : [0, 0] if (pd.isnull(hpy) + (hpy == 0)) else [mass/hpy, mass]

            return [conv(mass, row['2012']),
                    conv(mass, row['2013']),
                    conv(mass, row['2014']),
                    conv(mass, row['2015']),
                    conv(mass, row['2016']),
                    conv(mass, row['2017']),
                    conv(mass, row['2018']),
                    conv(mass, row['2019']),
                    conv(mass, row['2020']),
                    vessel_type]

        rename_dict = {'seine': 'seiners',
                       'trawl': 'trawlers_and_dredgers',
                       'dredge': 'trawlers_and_dredgers',
                       'set_longline': 'fixed_gear',
                       'set_gillnet': 'fixed_gear',
                       'pot': 'fixed_gear',
                       'fixed': 'fixed_gear',
                       'drifting_longline': 'drifting_longlines',
                       'jigg': 'squid_jigger',
                       'pole': 'pole_and_line_and_trollers',
                       'troll': 'pole_and_line_and_trollers'}

        for category in rename_dict.keys():
            if category in vessel_class:
                if not pd.isnull(row['LOA']):
                    return mass_loss_rate(row['LOA']*gear_intensity[category],
                                          row, rename_dict[category])
                else:
                    return null

        # Anything reaching this point is unclassified ('fishing')
        if not pd.isnull(row['LOA']):
            return mass_loss_rate(row['LOA'], row, 'fishing')
        else:
            return null


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
          'vessel_class']] = data.apply(lambda row: calculate_lost_gear_mass_per_hr(row), axis=1, result_type='expand')

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
        data.loc[i, 'vessel_class'] = mmsi_entry['vessel_class'].values[0]
        data.loc[i, ['mass_rate_A', 'mass_rate_B']] = [mmsi_entry[year].values[0][0]*hrs,
                                                       mmsi_entry[year].values[0][1]*hrs]

    # Now grid
    for i, vtype in enumerate(['seiners', 'trawlers_and_dredgers', 'fixed_gear',
                               'drifting_longlines', 'squid_jigger',
                               'pole_and_line_and_trollers', 'fishing', 'all']):

        if vtype == 'all':
            grid[i, :, :] += np.histogram2d(data['cell_ll_lon'],
                                            data['cell_ll_lat'],
                                            bins=[lon_bnd, lat_bnd],
                                            weights=data['mass_rate_A'])[0].T

            grid[i+8, :, :] += np.histogram2d(data['cell_ll_lon'],
                                            data['cell_ll_lat'],
                                            bins=[lon_bnd, lat_bnd],
                                            weights=data['mass_rate_B'])[0].T
        else:
            subdata = data[data['vessel_class'] == vtype]
            grid[i, :, :] += np.histogram2d(subdata['cell_ll_lon'],
                                            subdata['cell_ll_lat'],
                                            bins=[lon_bnd, lat_bnd],
                                            weights=subdata['mass_rate_A'])[0].T

            subdata = data[data['vessel_class'] == vtype]
            grid[i+8, :, :] += np.histogram2d(subdata['cell_ll_lon'],
                                            subdata['cell_ll_lat'],
                                            bins=[lon_bnd, lat_bnd],
                                            weights=subdata['mass_rate_B'])[0].T

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

    grid = np.zeros((16, len(lat), len(lon)), dtype=np.float64)

    # Generate MMSI dict
    try:
        mmsi_lookup = pd.read_pickle(fh['ref_proc'])
    except:
        mmsi_lookup = dictgen(fh['ref_raw'], fh['ref_proc'], gear_intensity)

    # Grid fishing effort
    for k, file in enumerate(fh['obs']):
        grid = grid_effort(file, grid, mmsi_lookup, lon_bnd, lat_bnd)
        print('{:.2f}'.format(100*k/len(fh['obs'])) + '%')


    # Plot output
    f0, ax = plt.subplots(2, 4, figsize=(20, 10),
                          subplot_kw={'projection': ccrs.PlateCarree()})
    f0.subplots_adjust(hspace=0.01, wspace=0.01, top=0.925, left=0.05)

    axpos = ax[-1, -1].get_position()
    pos_x = axpos.x0+axpos.width + 0.01
    pos_y = axpos.y0
    cax_width = 0.015
    cax_height = 2.01*axpos.height

    pos_cax = f0.add_axes([pos_x, pos_y, cax_width, cax_height])

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='black')

    text_labels = ['Seiners', 'Trawlers and dredgers', 'Fixed gear',
                   'Drifting longlines', 'Squid jiggers',
                   'Pole and line, and trollers', 'Unclassified', 'All']

    for i in range(8):
        axis = ax.flatten()[i]
        grid_data = grid[i, :, :]
        grid_data[np.isnan(grid_data)] = 0  # But why are NaNs appearing in the first place?
        grid_data_nonzero = np.copy(grid_data)
        grid_data_nonzero[grid_data == 0] = np.nan
        p05 = np.nanpercentile(grid_data_nonzero, 5)
        p95 = np.nanpercentile(grid_data_nonzero, 95)
        heatmap = axis.pcolormesh(lon_bnd, lat_bnd, grid_data/p95, cmap=cmr.chroma_r,
                                             norm=colors.LogNorm(vmin=1e-3, vmax=1))
        axis.add_feature(land_10m)
        axis.text(98, 26, text_labels[i], c='w', horizontalalignment='right')

        axis.axis('off')

    cb = plt.colorbar(heatmap, cax=pos_cax)
    cb.outline.set_visible(False)
