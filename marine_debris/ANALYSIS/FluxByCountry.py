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
         'cmap': cmr.guppy_r
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
      'data': sorted(glob(dirs['traj'] + 'data_s' + str(param['ls_d']) + '_b' + str(param['lb_d']) + '*.pkl')),}

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
    if data_fh != fh['data'][0]: # TEMPORARY TIME SAVING HACK, REMOVE!
        break

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

# Extract and normalise for presentation (choose <= 90% for each site)
country_prop_list = []
label_list = []
for i in range(len(data_by_sink)):
    data_i = data_by_sink[i].sort_values(axis=0)
    data_i = pd.DataFrame(data_i)
    total_flux = data_i['plastic_flux'].sum()

    data_i['all'] = pd.DataFrame(data_by_sink[-1])['plastic_flux']
    data_i['prop'] = data_i['plastic_flux']/data_i['plastic_flux'].sum()
    data_i['cumsum'] = np.cumsum(data_i['prop'][::-1])
    data_i=data_i.loc[data_i['cumsum'] <= param['cf_cutoff']]

    data_i = data_i.append(pd.DataFrame([[total_flux - data_i['plastic_flux'].sum(),
                                          0,
                                          1 - data_i['prop'].sum(),
                                          1]], columns=data_i.columns, index=[999]))

    label_list.append(data_i.index.values)
    country_prop_list.append(data_i.sort_values(['all'], axis=0, ascending=False))

# Get unique labels
label_list = np.unique(np.concatenate(label_list, axis=0))

# Make dict for names
source_dict = dict(zip(source_list['ISO code'], source_list['Country Name']))

# Calculate colormap
# Firstly sort labels by total mass
sorted_labels = np.array(data_by_sink[-1].filter(label_list).sort_values(ascending=False).index)
sorted_labels = np.concatenate([sorted_labels, [999]])
label_color = [param['cmap'](i) for i in np.linspace(0, param['cmap'].N, num=len(sorted_labels), dtype=int)]
color_dict = dict(zip(sorted_labels, label_color))

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
for source in sorted_labels:
    ax.bar(0, 0, 0, color=color_dict[source], label=source_dict[source])

cmap = param['cmap']
cmaplist = [cmap(i) for i in range(cmap.N)]

source_list_for_legend = []

for i in range(len(x_pos)):
    i_data = country_prop_list[i]
    cumsum = 0

    for j in range(len(i_data)):
        source = i_data.iloc[j].name

        ax.bar(x_pos[i], i_data['prop'].iloc[j], width, bottom=cumsum,
               color=color_dict[source])

        cumsum += i_data['prop'].iloc[j]

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

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=len(sorted_labels),
          frameon=False)

plt.savefig(fh['fig'], dpi=300)