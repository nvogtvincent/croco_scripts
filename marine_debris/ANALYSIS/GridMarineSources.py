#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot plastic fluxes to Seychelles by country and sink location
@author: Noam Vogt-Vincent
"""

### TO DO!!!!!
# Sources are currently excluded if they are not in the top X in a source, but are
# in others (should be included)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from glob import glob


##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = {# Analysis parameters
         'ls_d': 3650,      # Sinking timescale (days)
         'lb_d': 40,        # Beaching timescale (days)
         'title': 'Debris sources for zero windage, l(s)=10a, l(b)=20d',

         # Bounds
         'lon_west': 20,
         'lon_east': 130,
         'lat_south': -40,
         'lat_north': 30,
         }

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/../PLASTIC_DATA/',
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/../TRAJ/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'sink_list': dirs['plastic'] + 'sink_list.in',
      'grid':    dirs['grid'] + 'griddata.nc',
      'fig': dirs['fig'] + 'marine_plastic_sources_' + str(param['ls_d']) + '_b' + str(param['lb_d']),
      'data': sorted(glob(dirs['traj'] + 'marine_data_s' + str(param['ls_d']) + '_b' + str(param['lb_d']) + '*.pkl')),
      'cmap': dirs['fig'] + 'cmap_data.pkl'}

##############################################################################
# CALCULATE FLUXES                                                           #
##############################################################################

sink_list = pd.read_csv(fh['sink_list'])
nsink = np.shape(sink_list)[0]

# Grid accumulating fluxes by sink site
# Firstly construct grid
with Dataset(fh['grid'], mode='r') as nc:
    lon = nc.variables['lon_psi'][:]
    lat = nc.variables['lat_psi'][:]
    lon_bnd = np.concatenate([nc.variables['lon_rho'][:], [180]])
    lat_bnd = nc.variables['lat_rho'][:]

    lon = lon[(lon >= param['lon_west'])*(lon <= param['lon_east'])]
    lat = lat[(lat >= param['lat_south'])*(lat <= param['lat_north'])]
    lon_bnd = lon_bnd[(lon_bnd >= param['lon_west'])*(lon_bnd <= param['lon_east'])]
    lat_bnd = lat_bnd[(lat_bnd >= param['lat_south'])*(lat_bnd <= param['lat_north'])]

# Create empty grids for gridding
grid = np.zeros((nsink, len(lat), len(lon)))

for data_fh in fh['data']:
    # if data_fh != fh['data'][0]: # TEMPORARY TIME SAVING HACK, REMOVE!
    #     break

    data = pd.read_pickle(data_fh)
    print(data_fh)

    for sinkidx, sink in enumerate(sink_list['Sink code']):
        # Filter by sink site
        data_sink = data.loc[data['sink_id'] == sink]
        print(sinkidx)

        grid[sinkidx, :, :] += np.histogram2d(data_sink['lon0'].values,
                                              data_sink['lat0'].values,
                                              bins=[lon_bnd, lat_bnd],
                                              weights=data_sink['plastic_flux'].values)[0].T

##############################################################################
# PLOT                                                                       #
##############################################################################

f, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True,
                     subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(bottom=0, top=1)

data_crs = ccrs.PlateCarree()

mass_flux = ax.pcolormesh(lon_bnd, lat_bnd, np.sum(grid[:,:,:], axis=0)/(len(fh['data'])*1728),
                          norm=colors.LogNorm(vmin=1e-5, vmax=1e-2), cmap=cmr.torch, transform=data_crs)
# Multiplication by years * months per year * particles released per cell per month

ax.spines['geo'].set_linewidth(1)
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_aspect('equal')
# ax.set_title('Fraction of mass from cell beaching at Seychelles')

cax1 = f.add_axes([ax.get_position().x1+0.0105,ax.get_position().y0,0.015,ax.get_position().height])

cb1 = plt.colorbar(mass_flux, cax=cax1, pad=0.1)
cb1.set_label('Fraction of mass from cell beaching at Seychelles', size=12)

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='white', linewidth=0.5,
                                        facecolor='black')
ax.add_feature(land_10m)

gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='black', linestyle='-')
gl.xlocator = mticker.FixedLocator(np.arange(20, 140, 20))
gl.ylocator = mticker.FixedLocator(np.arange(-40, 40, 20))
gl.ylabels_right = False
gl.xlabels_top = False

plt.savefig(fh['fig'] + '_all.png', dpi=300, bbox_inches='tight')

###############################################################################

f, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True,
                     subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(bottom=0, top=1)

data_crs = ccrs.PlateCarree()

mass_flux = ax.pcolormesh(lon_bnd, lat_bnd, np.sum(grid[:4,:,:], axis=0)/(len(fh['data'])*1728),
                          norm=colors.LogNorm(vmin=1e-5, vmax=1e-2), cmap=cmr.torch, transform=data_crs)
# Multiplication by years * months per year * particles released per cell per month

ax.spines['geo'].set_linewidth(1)
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_aspect('equal')
# ax.set_title('Fraction of mass from cell beaching at Seychelles')

cax1 = f.add_axes([ax.get_position().x1+0.0105,ax.get_position().y0,0.015,ax.get_position().height])

cb1 = plt.colorbar(mass_flux, cax=cax1, pad=0.1)
cb1.set_label('Fraction of mass from cell beaching at the Aldabra Group', size=12)

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='white', linewidth=0.5,
                                        facecolor='black')
ax.add_feature(land_10m)

gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='black', linestyle='-')
gl.xlocator = mticker.FixedLocator(np.arange(20, 140, 20))
gl.ylocator = mticker.FixedLocator(np.arange(-40, 40, 20))
gl.ylabels_right = False
gl.xlabels_top = False

plt.savefig(fh['fig'] + '_AldabraGrp.png', dpi=300, bbox_inches='tight')

###############################################################################

f, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True,
                     subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(bottom=0, top=1)

data_crs = ccrs.PlateCarree()

mass_flux = ax.pcolormesh(lon_bnd, lat_bnd, np.sum(grid[12:,:,:], axis=0)/(len(fh['data'])*1728),
                          norm=colors.LogNorm(vmin=1e-5, vmax=1e-2), cmap=cmr.torch, transform=data_crs)
# Multiplication by years * months per year * particles released per cell per month

ax.spines['geo'].set_linewidth(1)
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_aspect('equal')
# ax.set_title('Fraction of mass from cell beaching at Seychelles')

cax1 = f.add_axes([ax.get_position().x1+0.0105,ax.get_position().y0,0.015,ax.get_position().height])

cb1 = plt.colorbar(mass_flux, cax=cax1, pad=0.1)
cb1.set_label('Fraction of mass from cell beaching at the Seychelles Plateau', size=12)

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='white', linewidth=0.5,
                                        facecolor='black')
ax.add_feature(land_10m)

gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='black', linestyle='-')
gl.xlocator = mticker.FixedLocator(np.arange(20, 140, 20))
gl.ylocator = mticker.FixedLocator(np.arange(-40, 40, 20))
gl.ylabels_right = False
gl.xlabels_top = False

plt.savefig(fh['fig'] + '_SeychellesPlateau.png', dpi=300, bbox_inches='tight')