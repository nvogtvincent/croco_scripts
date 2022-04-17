#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise marine debris results
@author: noam
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cmasher as cmr
import xarray as xr
from datetime import timedelta, datetime
from glob import glob
import time
from skimage.measure import block_reduce
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle

# PARAMETERS
param = {# Analysis parameters
         'us_d': 100,    # Sinking timescale (days)
         'ub_d': 20,      # Beaching timescale (days)

         # Time range
         'y0'  : 1993,
         'y1'  : 2012,

         # Physics
         'mode': '0000',

         # CMAP
         'cmap': cmr.guppy_r,
         'write_cmap': False, # Whether to write cmap data
         'n_source': 10,

         # Export
         'export': True}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/../PLASTIC_DATA/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'flux': dirs['script'] + '/terrestrial_flux_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'drift_time': dirs['script'] + '/terrestrial_drift_time_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'source_list': dirs['plastic'] + 'country_list.in',
      'sink_list': dirs['plastic'] + 'sink_list.in',
      'cmap': dirs['fig'] + 'cmap_data.pkl',
      'fig': dirs['fig'] + 'terrestrial_sources_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf'}

n_years = param['y1']-param['y0']+1
##############################################################################
# LOAD DATA                                                                  #
##############################################################################

fmatrix = xr.open_dataarray(fh['flux'])
dtmatrix = xr.open_dataarray(fh['drift_time'])

# Calculate the accumulated fluxes (integrating over all source and sink times)
fmatrix = fmatrix.sum(dim=('sink_time', 'source_time'))

##############################################################################
# PROCESS DATA                                                               #
##############################################################################

if param['write_cmap']:
    # Calculate the 10 most significant source countries (normalised per sink)
    # source_norm = np.array([fmatrix.loc[:, sink]/(fmatrix.sum(dim='source').loc[sink]) for sink in fmatrix.coords['sink']])
    fmatrix_norm = fmatrix/fmatrix.sum(dim='source')
    fmatrix_pop = fmatrix_norm.sum(dim='sink')
    fmatrix_pop = fmatrix_pop.sortby(fmatrix_pop, ascending=False)
    fmatrix_pop[fmatrix_pop['source'] == 'Other'] = fmatrix_pop[param['n_source']-2]-1e-6 # Hack to make sure 'other' is included
    fmatrix_pop = fmatrix_pop.sortby(fmatrix_pop, ascending=False)

    l1_source = fmatrix_pop[:param['n_source']]
    l2_source = fmatrix_pop[param['n_source']:]

    # # Now check that the top source for each sink is represented
    # for sink in fmatrix.coords['sink']:
    #     top_sources = fmatrix[:, fmatrix['sink'] == sink]
    #     top_source = top_sources.sortby(top_sources[:,0])[-1].coords['source']
    #     listy.append(top_source.values)
    #     if not top_source.isin(l1_source.coords['source']).values:
    #         print()

    other_flux = fmatrix[fmatrix['source'].isin(l2_source.coords['source'])].sum(dim='source')

    fmatrix = fmatrix[fmatrix['source'].isin(l1_source.coords['source'])]
    fmatrix[fmatrix['source'] == 'Other'] = other_flux
    fmatrix = fmatrix.sortby(fmatrix_pop, ascending=False)
    fmatrix = fmatrix/fmatrix.sum(dim='source')

    # Generate cmapping
    cmap_list = [param['cmap'](i) for i in np.linspace(0, param['cmap'].N, num=param['n_source'], dtype=int)]
    cmap_list = dict(zip(l1_source.coords['source'].values, cmap_list))

    with open(fh['cmap'], 'wb') as pkl:
        pickle.dump(cmap_list, pkl, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(fh['cmap'], 'rb') as pkl:
        cmap_list = pickle.load(pkl)

    l1_source = list(cmap_list.keys())
    other_flux = fmatrix[~fmatrix['source'].isin(l1_source)].sum(dim='source')

    fmatrix_norm = fmatrix/fmatrix.sum(dim='source')
    fmatrix_pop = fmatrix_norm.sum(dim='sink')
    fmatrix_pop = fmatrix_pop.sortby(fmatrix_pop, ascending=False)
    fmatrix_pop[fmatrix_pop['source'] == 'Other'] = fmatrix_pop[param['n_source']-2]-1e-6
    fmatrix_pop = fmatrix_pop.sortby(fmatrix_pop, ascending=False)
    fmatrix = fmatrix[fmatrix['source'].isin(l1_source)]
    fmatrix[fmatrix['source'] == 'Other'] = other_flux
    fmatrix = fmatrix/fmatrix.sum(dim='source')

    # Reorder
    fmatrix_order = fmatrix_pop.copy()
    fmatrix_order = fmatrix_order[fmatrix_order['source'].isin(l1_source)]
    for i in range(param['n_source']):
        source_name = l1_source[i]
        fmatrix_order[fmatrix_order['source'] == source_name] = i

    fmatrix = fmatrix.sortby(fmatrix_order)

##############################################################################
# PLOT                                                                       #
##############################################################################

f, ax = plt.subplots(1, 1, figsize=(40, 30), constrained_layout=True)

# Set up grouping
spacing = 1
width = 0.8

# Hack to get legend in the correct order
for j in range(param['n_source']):
    ax.bar(0, 0, 0, color=cmap_list[fmatrix.coords['source'].values[j]],
           label=fmatrix.coords['source'].values[j])

sites_per_group = np.array([4, 2, 1, 3, 2, 6, 9])
xpos = []
grp_mp = np.zeros_like(sites_per_group)

n_sink = len(fmatrix.coords['sink'])
assert n_sink == np.sum(sites_per_group)

grp, pos_in_grp = 0, -1
for i in range(len(fmatrix.coords['sink'])):
    xpos_ = xpos[-1] + spacing if i > 0 else spacing
    pos_in_grp += 1

    if pos_in_grp == sites_per_group[grp]:
        xpos_ += spacing
        pos_in_grp = 0
        grp += 1

    grp_mp[grp] += xpos_
    xpos.append(xpos_)

grp_mp = grp_mp/sites_per_group

cmap = param['cmap']
cmaplist = [cmap(i) for i in range(cmap.N)]

source_list_for_legend = []

for i in range(n_sink):
    cumsum = 0

    for j in range(param['n_source']):
        ax.bar(xpos[i], fmatrix[j, i], width, bottom=cumsum, color=cmap_list[fmatrix.coords['source'].values[j]])

        cumsum += fmatrix[j, i]

ax.set_xticks(grp_mp)
ax.set_xticklabels(['Aldabra Group', 'Farquhar Group', 'Alphonse Group',
                    'Amirante Islands', 'Southern Coral Group', 'Seychelles Plateau',
                    'Other'], fontsize=24)
ax.tick_params(axis='y', labelsize=28)

ax.set_ylim([0, 1])
ax.set_xlim([xpos[0]-width, xpos[-1]+width])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylabel('Proportion of beaching terrestrial debris from source', fontsize=36)
# ax.set_title(param['title'], y=1.02)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=param['n_source'],
          frameon=False, fontsize=28)

plt.savefig(fh['fig'], dpi=300)


