#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise marine debris results to compare physics cases
@author: noam
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import xarray as xr
import pickle
from skimage.measure import block_reduce
from glob import glob

##############################################################################
# PARAMETERS                                                                 #
##############################################################################

# PARAMETERS
param = {# Analysis parameters
         'us_d': 360.0,    # Sinking timescale (days)
         'ub_d': 30.0,    # Beaching timescale (days)
         'c_frac': 0.25, # Fraction of coastal plastics entering the ocean

         # Sink list
         'sink': ['Aldabra', 'Farquhar', 'Alphonse', 'Praslin', 'BIOT', 'Pemba', 'Lakshadweep'],

         # Time range
         'y0'  : 1993,
         'y1'  : 2014,

         # CMAP
         'cmap': cmr.guppy_r,
         'write_cmap': False, # Whether to write cmap data (good w/ 100/0010)
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
      'fig': dirs['fig'] + 'terrestrial_sources_physics_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf'}

n_years = param['y1']-param['y0']+1
n_sink = len(param['sink'])
physics_list = ['0000NS', '0000', '0010', '0020', '0030']

##############################################################################
# LOAD DATA                                                                  #
##############################################################################

# Loop through the five files
for i, fh_i in enumerate(sorted([dirs['script'] + '/terrestrial_flux_' + mode + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc' for mode in physics_list])):
    fmatrix = xr.open_dataarray(fh_i)

    # Calculate the accumulated fluxes (integrating over 1997-2014 assuming <=1y sink time)
    fmatrix = fmatrix.sum(dim=('source_time', 'sink_time'))

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
    for k in range(param['n_source']):
        source_name = l1_source[k]
        fmatrix_order[fmatrix_order['source'] == source_name] = k

    fmatrix = fmatrix.sortby(fmatrix_order)

    # Plot

    if i == 0:
        f, ax = plt.subplots(1, 1, figsize=(25, 15))

        # Hack to get legend in the correct order
        for j in range(param['n_source']):
            ax.bar(0, 0, 0, color=cmap_list[fmatrix.coords['source'].values[j]],
                    label=fmatrix.coords['source'].values[j])

        x_pos = []
        x_mp_pos = []
        x_cnt = 0
        spacing = 0.2
        width = 0.8

        for j in range(5*n_sink):

            if x_cnt == 5:
                x_cnt = 0
                x_pos.append(x_pos[-1] + 5*spacing + width)
            elif j == 0:
                x_pos.append(spacing + width)
            else:
                x_pos.append(x_pos[-1] + spacing + width)

            if x_cnt%5 == 0:
                ax.text(x_pos[-1], 1.01, 'C', ha='center', fontsize=16, fontweight='bold')
            elif x_cnt%5 == 1:
                ax.text(x_pos[-1], 1.01, 'CS0', ha='center', fontsize=16, fontweight='bold')
            elif x_cnt%5 == 2:
                ax.text(x_pos[-1], 1.01, 'CS1', ha='center', fontsize=16, fontweight='bold')
            elif x_cnt%5 == 3:
                ax.text(x_pos[-1], 1.01, 'CS2', ha='center', fontsize=16, fontweight='bold')
            else:
                ax.text(x_pos[-1], 1.01, 'CS3', ha='center', fontsize=16, fontweight='bold')

            x_cnt += 1

        for j in range(n_sink):
            x_mp_pos.append(np.mean(x_pos[j*5:(j+1)*5]))

        ax.set_xlim([0, x_pos[-1] + spacing + width])
        ax.set_ylim([0, 1])

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel('Proportion of terrestrial debris from source', fontsize=36)

        ax.set_xticks(x_mp_pos)
        x_labels = np.copy(param['sink'])
        x_labels[x_labels == 'BIOT'] = 'Chagos'
        ax.set_xticklabels(x_labels, fontsize=32)
        ax.tick_params(axis='y', labelsize=28)

        cmap = param['cmap']
        cmaplist = [cmap(l) for l in range(cmap.N)]

        source_list_for_legend = []

        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=int(param['n_source']/2),
                  frameon=False, fontsize=28)

    print()

    for sink_j in range(n_sink):
        cumsum = 0

        for source_k in range(param['n_source']):
            src_name = fmatrix.coords['source'].values[source_k]
            ax.bar(x_pos[sink_j*5+i], fmatrix.loc[src_name, param['sink'][sink_j]], width,
                   bottom=cumsum, color=cmap_list[src_name])
            cumsum += fmatrix.loc[src_name, param['sink'][sink_j]]

plt.savefig(fh['fig'], dpi=300, bbox_inches="tight")
