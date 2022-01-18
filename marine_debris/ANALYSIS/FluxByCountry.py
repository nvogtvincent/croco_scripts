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
from glob import glob
import matplotlib.pyplot as plt
import cmasher as cmr

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = {# Analysis parameters
         'ls_d': 3650,      # Sinking timescale (days)
         'lb_d': 20,        # Beaching timescale (days)
         'cf_cutoff': 0.95, # Cumulative frequency cutoff
         'title': 'Debris sources for zero windage, l(s)=10a, l(b)=20d',
         'cmap': cmr.guppy_r,
         'write_cmap': True # Whether to write cmap data
         }

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/../PLASTIC_DATA/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/../TRAJ/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'source_list': dirs['plastic'] + 'country_list.in',
      'sink_list': dirs['plastic'] + 'sink_list.in',
      'fig': dirs['fig'] + 'flux_by_country_s' + str(param['ls_d']) + '_b' + str(param['lb_d']) + '.png',
      'data': sorted(glob(dirs['traj'] + 'data_s' + str(param['ls_d']) + '_b' + str(param['lb_d']) + '*.pkl')),
      'cmap': dirs['fig'] + 'cmap_data.pkl'}

##############################################################################
# CALCULATE FLUXES                                                           #
##############################################################################

source_list = pd.read_csv(fh['source_list'])
source_list = source_list.append(pd.DataFrame([[999, 'Other']], columns=source_list.columns))
sink_list = pd.read_csv(fh['sink_list'])

nsource = np.shape(source_list)[0]
nsink = np.shape(sink_list)[0]

# First sort by sink site, then group by source location
data_by_sink = []

for data_fh in fh['data']:
    # if data_fh != fh['data'][0]: # TEMPORARY TIME SAVING HACK, REMOVE!
    #     break

    data = pd.read_pickle(data_fh)
    print(data_fh)

    for sinkidx, sink in enumerate(sink_list['Sink code']):

        # Filter by sink site
        data_sink = data.loc[data['sink_id'] == sink]

        # Group and sum by source
        data_sink = data_sink.groupby(['source_id']).sum()['plastic_flux']

        # Add to previous year
        if data_fh == fh['data'][0]:
            data_by_sink.append(data_sink.copy())
        else:
            data_by_sink[sinkidx] = data_by_sink[sinkidx].add(data_sink, fill_value=0)

    # All sites (total)
    data = data.groupby(['source_id']).sum()['plastic_flux']

    if data_fh == fh['data'][0]:
        data_by_sink.append(data.copy())
    else:
        data_by_sink[-1] = data_by_sink[-1].add(data, fill_value=0)

# On the basis of the combined mass flux, extract the top 10 countries
data_all = data_by_sink[-1].copy()
data_all = data_all.sort_values(axis=0, ascending=False)
data_all = np.array(data_all.index[:10])
data_all = np.concatenate([data_all, [999]])
country_order = pd.DataFrame({'source_id': data_all})

# Attribute colours and names
source_dict = dict(zip(source_list['ISO code'], source_list['Country Name']))

if param['write_cmap']:

    country_order['name'] = [source_dict[i] for i in country_order['source_id']]
    country_order['cmap'] = [param['cmap'](i) for i in np.linspace(0, param['cmap'].N, num=len(country_order), dtype=int)]
    country_order.to_pickle(fh['cmap'])

else:
    country_order = pd.read_pickle(fh['cmap'])

# Now extract the proportions of these countries from the data
country_prop_list = pd.DataFrame()
for i in range(len(data_by_sink)):
    data_i = data_by_sink[i]
    data_i_out = data_by_sink[i].loc[np.isin(data_by_sink[i].index,
                                             country_order['source_id'])]
    data_i_out[999] = data_by_sink[i].loc[~np.isin(data_by_sink[i].index,
                                                   country_order['source_id'])].sum()

    country_prop_list[i+1] = data_i_out/data_i_out.sum() # [i+1] because sink_id starts from 1

# Make dict for names
source_dict = dict(zip(source_list['ISO code'], source_list['Country Name']))

##############################################################################
# PLOT FLUXES                                                                #
##############################################################################
f, ax = plt.subplots(1, 1, figsize=(15, 10), constrained_layout=True)

# Set up grouping
spacing = 1.5
width = 0.8

sites_per_group = np.array([4, 2, 1, 3, 2, 6, 1])

x_cpos = np.cumsum(sites_per_group*width + spacing) - 0.5*sites_per_group*width
x_pos = np.repeat(x_cpos, sites_per_group)

grp = 0
pos_in_grp = 0

for i in range(len(x_pos)):
    num_in_grp = sites_per_group[grp]
    x_shift = pos_in_grp - ((num_in_grp-1)/2)

    x_pos[i] += x_shift

    pos_in_grp += 1
    if pos_in_grp == sites_per_group[grp]:
        grp += 1
        pos_in_grp = 0

# Plot data
# Firstly use a quick hack to get the legend in the correct order
for source in country_order['source_id']:
    ax.bar(0, 0, 0, color=country_order['cmap'].loc[country_order['source_id']==source],
           label=country_order['name'].loc[country_order['source_id']==source].values[0])

cmap = param['cmap']
cmaplist = [cmap(i) for i in range(cmap.N)]

source_list_for_legend = []

for i in range(len(x_pos)):
    cumsum = 0

    for j, source in enumerate(country_order['source_id']):
        ax.bar(x_pos[i], country_prop_list[i+1].loc[source], width,
               bottom=cumsum, color=country_order['cmap'].loc[country_order['source_id']==source])

        cumsum += country_prop_list[i+1].loc[source]

ax.set_xticks(x_cpos)
ax.set_xticklabels(['Aldabra Group', 'Farquhar Group', 'Alphonse Group',
                    'Amirante Islands', 'Southern Coral Group', 'Seychelles Plateau',
                    'All sites'])
ax.set_ylim([0, 1])
ax.set_xlim([x_pos[0]-width, x_pos[-1]+width])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylabel('Proportion of beaching terrestrial debris from source')
ax.set_title(param['title'], y=1.02)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=len(country_order),
          frameon=False)

plt.savefig(fh['fig'], dpi=300)