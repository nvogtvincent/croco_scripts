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
         'lb_d': 100,        # Beaching timescale (days)
         'title': 'Drift time for mass accumulation at Seychelles, l(s)=10a, l(b)=20d',
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
      'fig': dirs['fig'] + 'mass_accumulation_histogram_' + str(param['ls_d']) + '_b' + str(param['lb_d']) + '.png',
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

country_order = pd.read_pickle(fh['cmap'])

# Time bins
time_bnds = np.linspace(0, 9, num=121)
time = 0.5*(time_bnds[:-1] + time_bnds[1:])

# Create empty arrays to store histograms for each island
hist = np.zeros((len(country_order), len(time)), dtype=np.float64)

for data_fh in fh['data']:
    # if data_fh != fh['data'][0]: # TEMPORARY TIME SAVING HACK, REMOVE!
    #     break

    # Read data
    data = pd.read_pickle(data_fh)
    print(data_fh)

    # Now bin (weighted by mass) through the island sinks:
    for sourcei, source in enumerate(country_order['source_id']):
        if source == 999:
            subdata = data.loc[~np.isin(data['source_id'], country_order['source_id'])]
        else:
            subdata = data.loc[data['source_id'] == source]

        hist[sourcei, :] = np.histogram(subdata['days_at_sea']/365.25, bins=time_bnds,
                                        weights=subdata['plastic_flux'])[0]

# Normalise hist so integral is 1
hist /= np.sum(hist)*(time[1]-time[0])

##############################################################################
# PLOT FLUXES                                                                #
##############################################################################
f, ax = plt.subplots(1, 1, figsize=(15, 10), constrained_layout=True)

cumsum = np.zeros_like(time)

# Firstly use a quick hack to get the legend in the correct order
for source in country_order['source_id']:
    ax.bar(0, 0, 0, color=country_order['cmap'].loc[country_order['source_id']==source],
           label=country_order['name'].loc[country_order['source_id']==source].values[0])

for sourcei, source in enumerate(country_order['source_id']):
    ax.bar(time, hist[sourcei, :], width=time[1]-time[0], bottom=cumsum,
           color=country_order['cmap'].loc[country_order['source_id'] == source].values[0])
    cumsum += hist[sourcei, :]


ax.set_ylim([0, np.max(cumsum)])
ax.set_xlim([0, 10])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylabel('Probability density')
ax.set_xlabel('Time at sea (years)')
ax.set_title(param['title'], y=1.02)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(country_order),
          frameon=False)

plt.savefig(fh['fig'], dpi=300)
