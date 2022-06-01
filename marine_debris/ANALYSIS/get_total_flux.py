#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get total marine debris flux to location
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmasher as cmr
import xarray as xr
from scipy.interpolate import interp1d
import pickle

# PARAMETERS
param = {# Analysis parameters
         'us_d': 81.11308308,    # Sinking timescale (days)
         'ub_d': 25.0,    # Beaching timescale (days)
         'c_frac': 0.25, # Fraction of coastal plastics entering the ocean

         # Time range
         'y0'  : 1993,
         'y1'  : 2014,

         # Physics
         'mode': '0000',

         # Destination
         'destination': 'Aldabra',

         # CMAP
         'cmap': cmr.guppy_r,

         # Export
         'export': True}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/../PLASTIC_DATA/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'flux': dirs['script'] + '/terrestrial_flux_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'global_prod': dirs['plastic'] + '/Geyer2017/global-plastics-production.csv',
      'global_fate': dirs['plastic'] + '/Geyer2017/global-plastic-fate.csv',
      'cousine_accum': dirs['plastic'] + '/Dunlop2020/dunlop_plastic_accumulation.xlsx',
      'source_list': dirs['plastic'] + 'country_list.in',
      'sink_list': dirs['plastic'] + 'sink_list.in',
      'cmap': dirs['fig'] + 'cmap_data.pkl',
      'fig1': dirs['fig'] + 'land_sources_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf',
      'fig2': dirs['fig'] + 'land_drift_time_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf'}

##############################################################################
# LOAD DATA                                                                  #
##############################################################################

fmatrix = xr.open_dataarray(fh['flux'])

global_prod = pd.read_csv(fh['global_prod'])
global_fate = pd.read_csv(fh['global_fate'])

cousine_accum = pd.read_excel(fh['cousine_accum'], skiprows=8, usecols=[i for i in range(1,11)],
                              nrows=2)

# Calculate a scale factor to convert annual fluxes to accumulated flux
# (assuming no decay on-shore)

# Meijer 2021: Estimates are for 2015
# Lebreton 2019: Estimates are for 2015
# Burt 2020: Survey carried out in March 2019
# From model, average drift time for accumulating plastic at Aldabra is around 6
# months for month-scale sinking, and 18 months for 5-year scale sinking. So
# let's sum plastic fluxes for emissions up to Jun 2018.

global_fate = global_fate[global_fate['Entity'] == 'Discarded'][['Year', 'Estimated historic plastic fate']]
global_fate = global_fate.set_index('Year')
global_fate['Estimated historic plastic fate'] *= 1e-2
global_prod = global_prod.rename(columns={'Global plastics production (million tonnes)': 'production'})
global_prod = global_prod.set_index('Year')
global_prod['discarded_frac'] = global_fate['Estimated historic plastic fate']
global_prod = global_prod[['production', 'discarded_frac']]
global_prod_ = pd.DataFrame(np.array([np.arange(2016,2019, dtype=int), np.ones((3,))*global_prod.iloc[-1]['production'],
                             np.ones((3,))*global_prod.iloc[-1]['discarded_frac']]).T, columns=['Year', 'production', 'discarded_frac']).astype({'Year': 'int16'}).set_index('Year')
global_prod = pd.concat([global_prod, global_prod_])
global_prod = global_prod.fillna(1.0)
global_prod['discarded'] = global_prod['production']*global_prod['discarded_frac']

total_prod = global_prod.loc[:2017].sum() + global_prod.loc[2018]*0.5
total_prod = total_prod['discarded']

sf_prod = total_prod/global_prod.loc[2015]['discarded']

# Firstly, only consider debris arriving between 1999-01 to 2014-12 (to allow a
# 6-year spin-up), from all sources and source times, arriving at Aldabra
fmatrix_sub = fmatrix[:, :, :, 72:264].sum(dim=('source_time', 'source')).loc['Aldabra']

# Find sum
model_accumulation = fmatrix_sub.sum()

# Find accumulation per year
annual_flux = model_accumulation/len(np.unique(fmatrix_sub.sink_time.dt.year))

# Estimated total accumulation
total_accumulation_prod = sf_prod*annual_flux

# Also create a scale factor based on figures from Dunlop (Cousine)
# We have accumulation rate estimates from 2003 to 2019. Let's assume a similar
# trend holds true for Aldabra. Calculate the total estimated accumulation on
# Cousine (assuming 0 accumulation before first survey), and divide by the
# accumulation in 2015 to obtain a [total accumulation] to [accumulation in 2015]
# ratio. Then multiply by mean annual accumulation at Aldabra.

cousine_interp1 = interp1d(cousine_accum.iloc[0,:].values,
                           cousine_accum.iloc[1,:].values, kind='linear')

# Second case assuming pre-observation accumulation ramping down to 1950
cousine_interp2 = interp1d(np.insert(cousine_accum.iloc[0,:].values, 0, 1950),
                           np.insert(cousine_accum.iloc[1,:].values, 0, 0), kind='linear')

cousine_accum_interp1 = cousine_interp1(np.arange(2003, 2020))
cousine_accum_interp2 = cousine_interp2(np.arange(1950, 2020))

sf_cousine1 = np.sum(cousine_accum_interp1)/cousine_accum_interp1[-5]
sf_cousine2 = np.sum(cousine_accum_interp2)/cousine_accum_interp2[-5]

total_accumulation_cousine1 = sf_cousine1*annual_flux
total_accumulation_cousine2 = sf_cousine2*annual_flux

print('Estimated total terrestrial plastic accumulation on Aldabra (model extrapolation using global plastic production):')
print(str(np.round(total_accumulation_prod.values/1e3, decimals=1)) + ' tonnes')
print('')
print('Estimated total terrestrial plastic accumulation on Aldabra (model extrapolation using accumulation data at Cousine):')
print(str(np.round(total_accumulation_cousine1.values/1e3, decimals=1)) + ' tonnes')
print('')
print('Estimated total terrestrial plastic accumulation on Aldabra (model extrapolation using extrapolated accumulation data at Cousine):')
print(str(np.round(total_accumulation_cousine2.values/1e3, decimals=1)) + ' tonnes')
print('')
print('Estimated total terrestrial plastic accumulation on Aldabra (observations):')
print(str(np.round(513.4*0.17, decimals=1)) + ' tonnes')