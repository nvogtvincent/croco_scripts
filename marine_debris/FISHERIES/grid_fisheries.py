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


###############################################################################
# Parameters ##################################################################
###############################################################################

# PARAMETERS
param = {'grid_res': 1.0,                          # Grid resolution in degrees
         'lon_range': [20, 100],                 # Longitude range for output
         'lat_range': [-60, 30],                   # Latitude range for output

         'out_name': 'fisheries_waste_flux.png'}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'data_obs': os.path.dirname(os.path.realpath(__file__)) + '/DATA/OBS/',
        'data_ref': os.path.dirname(os.path.realpath(__file__)) + '/DATA/REF/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/FIGURES/'}

# FILE HANDLES
fh = {'obs': [],
      'ref': dirs['data_ref'] + 'fishing-vessels-v2.csv',
      'fig': dirs['fig'] + param['out_name']}

for path, subdirs, files in os.walk(dirs['data_obs']):
    for name in files:
        fh['obs'].append(os.path.join(path, name))

# GEAR INTENSITY & DISSIPATION RATE
dr = {0: 0.081,
      1: 0.014,
      2: 0.070,
      3: 0.024,
      4: 0.084}

gear_intensity = {0: {'LOA': 153.0*dr[0], # Purse seines (tuna)
                      'GT': 30.0*dr[0],
                      'kW': 16.5*dr[0]},
                  1: {'LOA': 153.0*dr[1], # Purse seines (non-tuna)
                      'GT': 30.0*dr[1],
                      'kW': 16.5*dr[1]},
                  2: {'LOA': 12.8*dr[2], # Trawlers
                      'GT': 8.3*dr[2],
                      'kW': 0.45*dr[2]},
                  3: {'LOA': 33.8*dr[3]}, # Longlines
                  4: {'dr': dr[4]}} # Gillnet
###############################################################################
# Functions ###################################################################
###############################################################################

# Generate MMSI dictionary
def dictgen(file, year, gear_intensity):
    # Load file
    data = pd.read_csv(file,
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

    # Calculate total gillnet fishing hours (for calculation)
    gillnet_sum = data['2020'][data['class'].str.contains('gill')].sum()

    # Assume that gillnet catch is proportional to fishing time
    # i.e. mass = 14.7kg/TC * (hours/total_hours) * total_catch
    #           = hours * (14.7kg/TC * total_catch/total_hours)
    #           = hours * (14.7kg/TC * 2.734*10^6 TC/3.095*10^6 h)
    #           = 13.0 kg/h (fishing equipment)
    # mass loss = loss_prop * mass/h

    gear_intensity[3]['h'] = gear_intensity[4]['dr']*(14.7*2.734*10**6)/gillnet_sum

    # Units of gear_intensity now: kg_lost/gear_dimension/yr (aside from gillnets which is kg_lost/hr)

    def calculate_lost_gear_mass_per_hr(row, gear_intensity):
        # Useful null list
        null = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Firstly establish whether vessel is applicable (return 0 if not)
        # if pd.isnull(row['time']) or row['time'] == 0:
        #     return null # lpy = mass loss per year

        # Establish the vessel type, then estimate the gear carried
        vessel_class = row['class']

        # Function to convert mass loss per year to array of mass loss per hour
        def lpy2lph(lpy, row):
            # Quick lambda that calculates loss per hour from loss per year and hours per year
            conv = lambda lpy, hpy : 0 if (pd.isnull(hpy) + (hpy == 0)) else lpy/hpy

            return [conv(lpy, row['2012']),
                    conv(lpy, row['2013']),
                    conv(lpy, row['2014']),
                    conv(lpy, row['2015']),
                    conv(lpy, row['2016']),
                    conv(lpy, row['2017']),
                    conv(lpy, row['2018']),
                    conv(lpy, row['2019']),
                    conv(lpy, row['2020']),]

        if "seine" in vessel_class:
            if ~pd.isnull(row['LOA']):
                lpy = row['LOA']*gear_intensity[0]['LOA']
            elif ~pd.isnull(row['GT']):
                lpy = row['GT']*gear_intensity[0]['GT']
            elif ~pd.isnull(row['kW']):
                lpy = row['kW']*gear_intensity[0]['kW']
            else:
                return null

            return lpy2lph(lpy, row)

        elif "trawl" in vessel_class or "dredge" in vessel_class:
            if ~pd.isnull(row['LOA']):
                lpy = row['LOA']*gear_intensity[1]['LOA']
            elif ~pd.isnull(row['GT']):
                lpy = row['GT']*gear_intensity[1]['GT']
            elif ~pd.isnull(row['kW']):
                lpy = row['kW']*gear_intensity[1]['kW']
            else:
                return null

            return lpy2lph(lpy, row)

        elif "longline" in vessel_class or "jigg" in vessel_class:
            if ~pd.isnull(row['LOA']):
                lpy = row['LOA']*gear_intensity[2]['LOA']
            else:
                return null

            return lpy2lph(lpy, row)

        elif "gill" in vessel_class:
            lpy = gear_intensity[3]['h']
            return [lpy, lpy, lpy, lpy, lpy, lpy, lpy, lpy, lpy]

        else:
            return null


    # Calculate mass of gear carried by vessel
    data[['loss_2012',
          'loss_2013',
          'loss_2014',
          'loss_2015',
          'loss_2016',
          'loss_2017',
          'loss_2018',
          'loss_2019',
          'loss_2020',]] = data.apply(lambda row: calculate_lost_gear_mass_per_hr(row, gear_intensity), axis=1, result_type='expand')

    return data[['mmsi', 'lost_gear_mass']]






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

    # Generate MMSI dict
    mmsi_lookup = dictgen(fh['ref'], 2018, gear_intensity)





    print()