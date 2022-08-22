#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New script to grid IOTC PS/LL effort data
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr
import pandas as pd
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec

###############################################################################
# Parameters ##################################################################
###############################################################################

# PARAMETERS
param = {'grid_res': 1.0,                          # Grid resolution in degrees
         'lon_range': [20, 130],                   # Longitude range for output
         'lat_range': [-40, 30]}                   # Latitude range for output


# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'data_obs': os.path.dirname(os.path.realpath(__file__)) + '/DATA/OBS/',
        'data_proc': os.path.dirname(os.path.realpath(__file__)) + '/DATA/PROC/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/FIGURES/'}

# FILE HANDLES
fh = {'ps': dirs['data_obs'] + 'IOTC-DATASETS-2022-02-23-CESurface_1950-2020.csv',
      'll': dirs['data_obs'] + 'IOTC-DATASETS-2022-02-23-CELongline_1950-2020.csv'}


###############################################################################
# Load data ###################################################################
###############################################################################

ps_df = pd.read_csv(fh['ps'], usecols=['Fleet', 'Gear', 'Year', 'MonthStart',
                                       'MonthEnd', 'Grid', 'Effort', 'EffortUnits',
                                       'QualityCode'])

ps_df = ps_df.loc[ps_df['Gear'].isin(['BBPS', 'PS', 'PSS', 'RIN', 'RNOF'])]
ps_df = ps_df.loc[ps_df['EffortUnits'].isin(['FDAYS', 'FHOURS'])]
ps_df = ps_df.loc[~ps_df['Fleet'].isin(['MYS    ', 'THA    '])]
ps_df['Effort'].loc[ps_df['EffortUnits'] == 'FDAYS'] *= 24 # All units to hours
ps_df = ps_df.loc[ps_df['MonthStart'] - ps_df['MonthEnd'] == 0]

ll_df = pd.read_csv(fh['ll'], usecols=['Fleet', 'Gear', 'Year', 'MonthStart',
                                       'MonthEnd', 'Grid', 'Effort', 'EffortUnits',
                                       'QualityCode'])
ll_df = ll_df.loc[ll_df['EffortUnits'].isin(['HOOKS'])]
ll_df = ll_df.loc[ll_df['MonthStart'] - ll_df['MonthEnd'] == 0]
ll_df = ll_df.loc[ll_df['Fleet'].isin(['JPN    ', 'KOR    ', 'TWN    '])]
ll_df = ll_df.loc[ll_df['QualityCode'] > 1]

###############################################################################
# Grid data ###################################################################
###############################################################################

# Generate grids
lon_bnd_ps = np.linspace(param['lon_range'][0],
                         param['lon_range'][1],
                         int((param['lon_range'][1] -
                              param['lon_range'][0]))+1)

lat_bnd_ps = np.linspace(param['lat_range'][0],
                         param['lat_range'][1],
                         int((param['lat_range'][1] -
                              param['lat_range'][0]))+1)

lon_ps = 0.5*(lon_bnd_ps[1:] + lon_bnd_ps[:-1])
lat_ps = 0.5*(lat_bnd_ps[1:] + lat_bnd_ps[:-1])

# Note: extend LL domain by 5 degrees for later upscaling
lon_bnd_ll = np.linspace(param['lon_range'][0]-5,
                         param['lon_range'][1]+5,
                         int((param['lon_range'][1] -
                              param['lon_range'][0])/5)+3)

lat_bnd_ll = np.linspace(param['lat_range'][0]-5,
                         param['lat_range'][1]+5,
                         int((param['lat_range'][1] -
                              param['lat_range'][0])/5)+3)

lon_ll = 0.5*(lon_bnd_ll[1:] + lon_bnd_ll[:-1])
lat_ll = 0.5*(lat_bnd_ll[1:] + lat_bnd_ll[:-1])

mon_bnd = np.arange(0.5, 13.5, 1)
mon = np.arange(1, 13, 1)

ps_grid = np.zeros((len(mon), len(lat_ps), len(lon_ps)), dtype=np.float32)
ll_grid = np.zeros((len(mon), len(lat_ll), len(lon_ll)), dtype=np.float32)

# Convert codes to lat/lon
def code2coord(code):
    adj_lon = np.zeros_like(code, dtype=float)
    adj_lat = np.zeros_like(code, dtype=float)
    lon_list = np.zeros_like(code, dtype=float)
    lat_list = np.zeros_like(code, dtype=float)

    # if len(code) > 0:
    adj_lon[(code.str[1] == '1')*(code.str[0] == '5')] = 0.5
    adj_lat[(code.str[1] == '1')*(code.str[0] == '5')] = 0.5
    adj_lon[(code.str[1] == '2')*(code.str[0] == '5')] = 0.5
    adj_lat[(code.str[1] == '2')*(code.str[0] == '5')] = -0.5
    adj_lon[(code.str[1] == '3')*(code.str[0] == '5')] = -0.5
    adj_lat[(code.str[1] == '3')*(code.str[0] == '5')] = -0.5
    adj_lon[(code.str[1] == '4')*(code.str[0] == '5')] = -0.5
    adj_lat[(code.str[1] == '4')*(code.str[0] == '5')] = 0.5
    adj_lon[(code.str[1] == '1')*(code.str[0] == '6')] = 2.5
    adj_lat[(code.str[1] == '1')*(code.str[0] == '6')] = 2.5
    adj_lon[(code.str[1] == '2')*(code.str[0] == '6')] = 2.5
    adj_lat[(code.str[1] == '2')*(code.str[0] == '6')] = -2.5
    adj_lon[(code.str[1] == '3')*(code.str[0] == '6')] = -2.5
    adj_lat[(code.str[1] == '3')*(code.str[0] == '6')] = -2.5
    adj_lon[(code.str[1] == '4')*(code.str[0] == '6')] = -2.5
    adj_lat[(code.str[1] == '4')*(code.str[0] == '6')] = 2.5

    lat_list[code.str[1] == '1'] = code[code.str[1] == '1'].str[2:4].astype(float) + adj_lat[code.str[1] == '1']
    lat_list[code.str[1] == '2'] = -code[code.str[1] == '2'].str[2:4].astype(float) + adj_lat[code.str[1] == '2']
    lat_list[code.str[1] == '3'] = -code[code.str[1] == '3'].str[2:4].astype(float) + adj_lat[code.str[1] == '3']
    lat_list[code.str[1] == '4'] = code[code.str[1] == '4'].str[2:4].astype(float) + adj_lat[code.str[1] == '4']
    lon_list[code.str[1] == '1'] = code[code.str[1] == '1'].str[4:].astype(float) + adj_lon[code.str[1] == '1']
    lon_list[code.str[1] == '2'] = code[code.str[1] == '2'].str[4:].astype(float) + adj_lon[code.str[1] == '2']
    lon_list[code.str[1] == '3'] = -code[code.str[1] == '3'].str[4:].astype(float) + adj_lon[code.str[1] == '3']
    lon_list[code.str[1] == '4'] = -code[code.str[1] == '4'].str[4:].astype(float) + adj_lon[code.str[1] == '4']

    return lon_list, lat_list

lon_list_ll, lat_list_ll = code2coord(ll_df['Grid'].astype('str'))
lon_list_ps, lat_list_ps = code2coord(ps_df['Grid'].astype('str'))

# Finally, filter data to limit within domain
ll_df['lon'] = lon_list_ll
ll_df['lat'] = lat_list_ll
ps_df['lon'] = lon_list_ps
ps_df['lat'] = lat_list_ps

# Grid
ll_grid = np.histogramdd([ll_df['MonthStart'].values, lat_list_ll, lon_list_ll],
                         bins=[mon_bnd, lat_bnd_ll, lon_bnd_ll],
                         weights=ll_df['Effort'], density=False)[0]
ps_grid = np.histogramdd([ps_df['MonthStart'].values, lat_list_ps, lon_list_ps],
                         bins=[mon_bnd, lat_bnd_ps, lon_bnd_ps],
                         weights=ps_df['Effort'], density=False)[0]

ll = xr.Dataset(data_vars=dict(effort=(['month', 'lat', 'lon'], ll_grid),
                               lon_bnd=(['lon_bnd'], lon_bnd_ll),
                               lat_bnd=(['lat_bnd'], lat_bnd_ll)),
                coords=dict(month=(['month'], mon),
                            lat=(['lat'], lat_ll),
                            lon=(['lon'], lon_ll)),
                attrs=dict(vessel_type='Longliner'))

ps = xr.Dataset(data_vars=dict(effort=(['month', 'lat', 'lon'], ps_grid),
                               lon_bnd=(['lon_bnd'], lon_bnd_ps),
                               lat_bnd=(['lat_bnd'], lat_bnd_ps)),
                coords=dict(month=(['month'], mon),
                            lat=(['lat'], lat_ps),
                            lon=(['lon'], lon_ps)),
                attrs=dict(vessel_type='Purse-seiner'))

ll = ll.interp_like(ps, method='nearest')

###############################################################################
# Plot data ###################################################################
###############################################################################


f = plt.figure(figsize=(17, 20), constrained_layout=True)
gs = GridSpec(2, 2, figure=f, width_ratios=[0.98, 0.03])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())) # Main figure top (PS)
ax.append(f.add_subplot(gs[0, 1])) # Colorbar
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree())) # Main figure bottom (LL)
ax.append(f.add_subplot(gs[1, 1])) # Colorbar

land_10k = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='k',
                                        facecolor='k',
                                        zorder=1)

ps_plot = ax[0].pcolormesh(ps.lon_bnd, ps.lat_bnd, ps.effort.mean(dim='month'),
                           transform=ccrs.PlateCarree(), norm=colors.LogNorm(vmin=1e0, vmax=1e4),
                           cmap=cmr.sunburst_r, zorder=1, rasterized=True)
ax[0].add_feature(land_10k)
ax[0].set_title('Purse-seine effort (IOTC)', size=32)
gl1 = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='k', linestyle='-', zorder=11)
gl1.xlocator = mticker.FixedLocator(np.arange(0, 225, 30))
gl1.ylocator = mticker.FixedLocator(np.arange(-60, 75, 30))
gl1.right_labels = False
gl1.top_labels = False
gl1.ylabel_style = {'size': 22}
gl1.xlabel_style = {'size': 22}

cb0 = plt.colorbar(ps_plot, cax=ax[1], fraction=0.1)
ax[1].tick_params(axis='y', labelsize=22)
cb0.set_label('Total effort (fishing-hours)', size=24)

###

ll_plot = ax[2].pcolormesh(ll.lon_bnd, ll.lat_bnd, ll.effort.loc[7,:,:],
                           transform=ccrs.PlateCarree(), norm=colors.LogNorm(vmin=1e4, vmax=1e8),
                           cmap=cmr.sunburst_r, zorder=1, rasterized=True)
ax[2].add_feature(land_10k)
ax[2].set_title('Longline effort (IOTC)', size=32)
ax[2].set_extent([20, 130, -40, 30], crs=ccrs.PlateCarree())

gl2 = ax[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='k', linestyle='-', zorder=11)
gl2.xlocator = mticker.FixedLocator(np.arange(0, 225, 30))
gl2.ylocator = mticker.FixedLocator(np.arange(-60, 75, 30))
gl2.right_labels = False
gl2.top_labels = False
gl2.ylabel_style = {'size': 22}
gl2.xlabel_style = {'size': 22}

cb1 = plt.colorbar(ll_plot, cax=ax[3], fraction=0.1)
ax[3].tick_params(axis='y', labelsize=22)
cb1.set_label('Total effort (hooks)', size=24)

ll.to_netcdf(dirs['data_proc'] + 'LL_IOTC_monclim.nc')
ps.to_netcdf(dirs['data_proc'] + 'PS_IOTC_monclim.nc')

plt.savefig(dirs['fig'] + 'IOTC_gridded.pdf', bbox_inches='tight', dpi=300)
