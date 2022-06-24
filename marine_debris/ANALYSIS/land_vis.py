#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise marine debris results
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
         'us_d': 360.0,    # Sinking timescale (days)
         'ub_d': 30.0,      # Beaching timescale (days)

         # Physics
         'mode': '0030',

         # Name
         'name': 'Class C',

         # CMAP
         'cmap': cmr.guppy_r,
         'write_cmap': False, # Whether to write cmap data (good w/ 100/0010)
         'n_source': 10,

         # Export
         'legend': True,
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
      'fig1': dirs['fig'] + 'terrestrial_sources_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf',
      'fig2': dirs['fig'] + 'terrestrial_drift_time_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf'}


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

    # Now repeat for dtmatrix
    other_dt = dtmatrix[dtmatrix['source'].isin(l2_source.coords['source'])].sum(dim='source')
    dtmatrix = dtmatrix[dtmatrix['source'].isin(l1_source.coords['source'])]
    dtmatrix[dtmatrix['source'] == 'Other'] = other_dt
    dtmatrix = dtmatrix.sortby(fmatrix_pop, ascending=False)
    dtmatrix = dtmatrix/dtmatrix.sum(dim=('source', 'drift_time'))

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

    # fmatrix_norm.loc[:, 'Reunion'].sortby(fmatrix_norm.loc[:, 'Reunion'], ascending=False).coords['source']
    # Reorder
    fmatrix_order = fmatrix_pop.copy()
    fmatrix_order = fmatrix_order[fmatrix_order['source'].isin(l1_source)]
    for i in range(param['n_source']):
        source_name = l1_source[i]
        fmatrix_order[fmatrix_order['source'] == source_name] = i

    fmatrix = fmatrix.sortby(fmatrix_order)

    # Repeat for dtmatrix
    other_dt = dtmatrix[~dtmatrix['source'].isin(l1_source)].sum(dim='source')
    dtmatrix = dtmatrix[dtmatrix['source'].isin(l1_source)]
    dtmatrix[dtmatrix['source'] == 'Other'] = other_dt
    dtmatrix = dtmatrix.sortby(fmatrix_order, ascending=True)
    dtmatrix = dtmatrix/dtmatrix.sum(dim=('source', 'drift_time'))

##############################################################################
# PLOT                                                                       #
##############################################################################

if param['legend']:
    f, ax = plt.subplots(1, 1, figsize=(20, 8))
else:
    f, ax = plt.subplots(1, 1, figsize=(20, 8))

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

grp_mp = np.concatenate((grp_mp[:-1], xpos[-9:]))

ax.set_xticks(grp_mp)
ax.set_xticklabels(['ALD', 'FAR', 'ALP',
                    'AMI', 'SCG', 'SEY',
                    '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize=24)
ax.tick_params(axis='y', labelsize=24)

ax.set_ylim([0, 1])
ax.set_xlim([xpos[0]-width, xpos[-1]+width])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylabel('Proportion of terrestrial debris', fontsize=24)
ax.set_title(param['name'] + ' debris sources', fontsize=28, color='k', fontweight='bold', pad=12)

if param['legend']:
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=int(param['n_source']/2),
              frameon=False, fontsize=24)

plt.savefig(fh['fig1'], dpi=300, bbox_inches="tight")

# Now plot a mass-time histogram for Aldabra only
reduction_ratio = 2
dpy = 365
new_time_axis = block_reduce(dtmatrix.coords['drift_time'], (reduction_ratio,), func=np.mean)/365
width = new_time_axis[1] - new_time_axis[0]
site_chosen = 'Alphonse'
f, ax = plt.subplots(1, 1, figsize=(40, 7), constrained_layout=True)
cumsum = np.zeros_like(block_reduce(dtmatrix.loc[dtmatrix.coords['source'].values[j], site_chosen, :], block_size=(reduction_ratio,), func=np.sum))

# Hack to get legend in the correct order
for j in range(param['n_source']):
    ax.bar(0, 0, 0, color=cmap_list[fmatrix.coords['source'].values[j]],
           label=fmatrix.coords['source'].values[j])

for j in range(param['n_source']):
    ax.bar(new_time_axis,
           block_reduce(dtmatrix.loc[dtmatrix.coords['source'].values[j], site_chosen, :], (reduction_ratio,), func=np.sum),
           width, bottom=cumsum, color=cmap_list[dtmatrix.coords['source'].values[j]])
    cumsum += block_reduce(dtmatrix.loc[dtmatrix.coords['source'].values[j], site_chosen, :], (reduction_ratio,), func=np.sum)

ax.set_xlim([0, 5])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# ax.set_ylabel('Proportion of terrestrial debris from source', fontsize=36)
ax.set_xlabel('Drifting time (years)', fontsize=24)
ax.tick_params(axis='x', labelsize=20)
ax.set_yticklabels([])
# ax.legend(loc="lower center", bbox_to_anchor=(0.5, -1), ncol=param['n_source'],
#           frameon=False, fontsize=28)
plt.savefig(fh['fig2'], dpi=300, bbox_inches="tight")
