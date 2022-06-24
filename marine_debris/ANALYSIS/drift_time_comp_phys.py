#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise marine debris results (compare drift time for different drift times)
@author: noam
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import xarray as xr
from skimage.measure import block_reduce
import pickle

# PARAMETERS
param = {# Analysis parameters
         'ub_d': 90.0,      # Beaching timescale (days)
         'ub_s': 30.0,     # Sinking timescale (days)

         # Site
         'sink': 'Aldabra',

         # CMAP
         'cmap': cmr.guppy_r,
         'write_cmap': True, # Whether to write cmap data (good w/ 100/0010)
         'n_source': 10,

         # Export
         'export': True}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/../PLASTIC_DATA/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'source_list': dirs['plastic'] + 'country_list.in',
      'sink_list': dirs['plastic'] + 'sink_list.in',
      'cmap': dirs['fig'] + 'cmap_data.pkl',
      'fig': dirs['fig'] + 'drift_time_comparison_physics_' + param['sink'] + '_s' + str(param['ub_s']) + '_b' + str(param['ub_d']) + '.pdf'}

mode_list = ['0000NS', '0000', '0010', '0020', '0030']
mode_name_list = ['C', 'CS0', 'CS1', 'CS2', 'CS3']

##############################################################################
# PROCESS DATA                                                               #
##############################################################################
for i in range(len(mode_list)):

# for i, fh_i in enumerate([dirs['script'] + '/terrestrial_flux_' + param['mode'] + '_s' + str(s) + '_b' + str(param['ub_d']) + '.nc' for s in us_list]):
    fmatrix = xr.open_dataarray(dirs['script'] + '/terrestrial_flux_' + mode_list[i] + '_s' + str(param['ub_s']) + '_b' + str(param['ub_d']) + '.nc')
    dtmatrix = xr.open_dataarray(dirs['script'] + '/terrestrial_drift_time_' + mode_list[i] + '_s' + str(param['ub_s']) + '_b' + str(param['ub_d']) + '.nc')

    # Calculate the accumulated fluxes (integrating over all source and sink times)
    fmatrix = fmatrix.sum(dim=('sink_time', 'source_time'))

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
    for j in range(param['n_source']):
        source_name = l1_source[j]
        fmatrix_order[fmatrix_order['source'] == source_name] = j

    fmatrix = fmatrix.sortby(fmatrix_order)

    # Repeat for dtmatrix
    other_dt = dtmatrix[~dtmatrix['source'].isin(l1_source)].sum(dim='source')
    dtmatrix = dtmatrix[dtmatrix['source'].isin(l1_source)]
    dtmatrix[dtmatrix['source'] == 'Other'] = other_dt
    dtmatrix = dtmatrix.sortby(fmatrix_order, ascending=True)/1e3

    ##############################################################################
    # PLOT                                                                       #
    ##############################################################################

    if i == 0:
        f, ax = plt.subplots(5, 1, figsize=(20, 20))

        dpy = 365
        reduction_ratio = 1
        new_time_axis = block_reduce(dtmatrix.coords['drift_time'], (reduction_ratio,), func=np.mean)/30
        width = new_time_axis[1] - new_time_axis[0]

    cumsum = np.zeros_like(block_reduce(dtmatrix.loc[dtmatrix.coords['source'].values[0], param['sink'], :], block_size=(reduction_ratio,), func=np.sum))

    # Hack to get legend in the correct order
    for j in range(param['n_source']):
        ax[i].bar(0, 0, 0, color=cmap_list[fmatrix.coords['source'].values[j]],
               label=fmatrix.coords['source'].values[j])

    for j in range(param['n_source']):
        ax[i].bar(new_time_axis,
                  block_reduce(dtmatrix.loc[dtmatrix.coords['source'].values[j], param['sink'], :], (reduction_ratio,), func=np.sum),
                  width, bottom=cumsum, color=cmap_list[dtmatrix.coords['source'].values[j]])
        cumsum += block_reduce(dtmatrix.loc[dtmatrix.coords['source'].values[j], param['sink'], :], (reduction_ratio,), func=np.sum)

    ax[i].set_xlim([0, 24])
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    if i  == len(mode_list) -1:
        ax[i].set_xlabel('Drifting time (months)', fontsize=24)
        ax[i].legend(loc="lower center", bbox_to_anchor=(0.5, -1.0), ncol=int(param['n_source']/2),
                     frameon=False, fontsize=24)

    ax[i].set_xticks(np.arange(25))
    ax[i].tick_params(axis='x', labelsize=22)
    ax[i].tick_params(axis='y', labelsize=22)
    ax[i].text(24, np.max(cumsum), mode_name_list[i],
               ha='right', va='top', fontsize=22)

plt.savefig(fh['fig'], dpi=300, bbox_inches="tight")
