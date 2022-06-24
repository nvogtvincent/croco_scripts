#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse time-series of marine debris accumulation at sites of interest
@author: noam
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import pandas as pd
import cmasher as cm
from scipy import signal
from datetime import datetime
from matplotlib.gridspec import GridSpec

# PARAMETERS
param = {# Analysis parameters
         'us_d': [30.0, 90.0, 360.0],  # Sinking timescale (days)
         'ub_d': [30.0, 30.0, 30.0],   # Beaching timescale (days)
         'physics': ['0000', '0010', '0030'],
         'name': ['Class A', 'Class B', 'Class C'],

         # Sink
         'site_list': ['Aldabra', 'Assomption', 'Cosmoledo', 'Astove', 'Providence',
                'Farquhar', 'Alphonse', 'Poivre', 'St Joseph', 'Desroches', 'Platte',
                'Coëtivy', 'Mahé', 'Fregate', 'Silhouette', 'Praslin', 'Denis', 'Bird',
                'Comoros', 'Mayotte', 'Lakshadweep', 'Maldives', 'Mauritius', 'Reunion',
                'Pemba', 'Socotra', 'BIOT'],

         'site_name': ['Aldabra', 'Assomption', 'Cosmoledo', 'Astove', 'Providence',
                'Farquhar', 'Alphonse', 'Poivre', 'St Joseph', 'Desroches', 'Platte',
                'Coëtivy', 'Mahé', 'Fregate', 'Silhouette', 'Praslin', 'Denis', 'Bird',
                'Comoros', 'Mayotte', 'Lakshadweep', 'Maldives', 'Mauritius', 'Reunion',
                'Pemba', 'Socotra', 'Chagos'],

         # Modes for comparison
         'mode': 'DMI',

         # Filtering parameters
         'lead': 0, # Lead (months) for climate index, (i.e. +ve -> index leads observations)
         'crit_freq': 12/16,

         # Remove 97/98
         'remove97': True,

         # Log threshold to avoid NaNs (kg)
         'threshold': 1e-9,}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'ref': os.path.dirname(os.path.realpath(__file__)) + '/../REFERENCE/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'DMI': dirs['ref'] + 'dmi.had.long.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/
      'NINO4': dirs['ref'] + 'nino4.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino4/
      'NINO34': dirs['ref'] + 'nino34.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino34/
      'NINO3': dirs['ref'] + 'nino3.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino3/
      'critical_values': dirs['ref'] + 'critical_values.txt'}

if param['remove97']:
    fh['fig'] = dirs['fig'] + 'time_series_' + param['mode'] + '_correlations_no9798.pdf'
else:
    fh['fig'] = dirs['fig'] + 'time_series_' + param['mode'] + '_correlations.pdf'

dof_mult = {'NINO4': 8, 'NINO34': 7, 'NINO3': 7, 'DMI': 4}
index_name = {'NINO4': 'NINO4', 'NINO3': 'NINO3', 'NINO34': 'NINO3.4', 'DMI': 'DMI'}

##############################################################################
# GENERATE OUTPUT MATRICES                                                   #
##############################################################################

lead_idx = np.arange(-24, 25, 1)
n_lead = len(lead_idx)
n_site = len(param['site_list'])
n_class = len(param['name'])

corr = []

for class_name in param['name']:
    corr.append(xr.DataArray(np.zeros((n_site, n_lead), dtype=np.float32),
                             coords={'site': param['site_list'],
                                     'lead': lead_idx},
                             name=class_name))

##############################################################################
# EXTRACT LAGGED CLIMATE INDICES                                             #
##############################################################################

n_month = 240
i0_fh = dirs['script'] + '/terrestrial_flux_' + param['physics'][0] + '_s' + str(param['us_d'][0]) + '_b' + str(param['ub_d'][0]) + '.nc'
time_idx = xr.open_dataarray(i0_fh)[:, :, :, 24:264].coords['sink_time']

if param['remove97']:
    time_idx = time_idx.where((time_idx.coords['sink_time'].dt.year != 1997)*(time_idx.coords['sink_time'].dt.year != 1998), drop=True)

clim_idx = xr.DataArray(np.zeros((n_lead, len(time_idx)), dtype=np.float32),
                        coords={'lead': lead_idx,
                                'sink_time': time_idx.values})

raw_clim = pd.read_table(fh[param['mode']], skiprows=1, delim_whitespace=True,
                         header=None, index_col=0, skipfooter=8, engine='python')
clim_subset = raw_clim[raw_clim.index.isin(np.arange(1995-3,2015+3))].values.flatten()
clim_subset = xr.DataArray(clim_subset, coords={'sink_time': pd.date_range(start=datetime(year=1992, month=1, day=1),
                                                                           end=datetime(year=2018, month=1, day=1),
                                                                           freq='M')})

if param['remove97']:
    clim_subset = clim_subset.where((clim_subset.coords['sink_time'].dt.year != 1997)*(clim_subset.coords['sink_time'].dt.year != 1998))

for i, lead in enumerate(lead_idx):
    clim_subset_lead = clim_subset[36-lead:-36-lead]
    clim_idx[i, :] = clim_subset_lead.dropna(dim='sink_time').values - np.mean(clim_subset_lead).values

# Calculate critical r value for significance
critical_table = pd.read_table(fh['critical_values'], delim_whitespace=True,
                               index_col=0, skipfooter=1, engine='python').loc[:, '0.05']
r_crit = critical_table[int(np.floor(len(time_idx)/(2*dof_mult[param['mode']])))]

##############################################################################
# LOOP THROUGH SITES AND CARRY OUT CORRELATION ANALYSIS                      #
##############################################################################

# Create low-pass filter
lp = signal.butter(3, param['crit_freq'], fs=12, btype='lowpass', analog=False, output='sos')

for i, class_name in enumerate(param['name']):
    md_fh = dirs['script'] + '/terrestrial_flux_' + param['physics'][i] + '_s' + str(param['us_d'][i]) + '_b' + str(param['ub_d'][i]) + '.nc'

    for site in param['site_list']:
        # Load (log) time-series
        beaching_ts = xr.open_dataarray(md_fh)[:, :, :, 24:264].sum(dim=('source_time', 'source')).loc[site, :]

        beaching_ts = beaching_ts.where(beaching_ts >= param['threshold'])
        beaching_ts = beaching_ts.fillna(param['threshold'])
        beaching_ts = np.log10(beaching_ts)

        # Pass through low-pass filter
        beaching_ts_lp = beaching_ts.copy()
        beaching_ts_lp.values = signal.sosfiltfilt(lp, beaching_ts_lp.values, axis=0)
        beaching_ts_lp -= beaching_ts_lp.mean()

        if param['remove97']:
            beaching_ts_lp = beaching_ts_lp.where((beaching_ts_lp.coords['sink_time'].dt.year != 1997)*(beaching_ts_lp.coords['sink_time'].dt.year != 1998),
                                                  drop=True)

        # Carry out correlations
        corr[i].loc[site, :] = xr.corr(beaching_ts_lp, clim_idx, dim='sink_time')

##############################################################################
# PLOT LAGGED CORRELATIONS                                                   #
##############################################################################

f = plt.figure(constrained_layout=True, figsize=(15, 20))
gs = GridSpec(3, 3, figure=f, width_ratios=[0.98, 0.02, 0.02])
ax = []
ax.append(f.add_subplot(gs[0, 0])) # Class A
ax.append(f.add_subplot(gs[1, 0])) # Class B
ax.append(f.add_subplot(gs[2, 0])) # Class C
ax.append(f.add_subplot(gs[1, 1])) # Colorbar 1
ax.append(f.add_subplot(gs[1, 2])) # Colorbar 2

corr_vis1 = []
corr_vis2 = []

# Plot correlations grids
for i, class_name in enumerate(param['name']):
    corr_vis1.append(ax[i].pcolormesh(np.append(lead_idx, lead_idx[-1]+1)-0.5, np.arange(n_site+1), corr[i].where(corr[i] < r_crit),
                                      cmap=cm.neutral_r, vmin=0, vmax=1))
    corr_vis2.append(ax[i].pcolormesh(np.append(lead_idx, lead_idx[-1]+1)-0.5, np.arange(n_site+1), corr[i].where(corr[i] >= r_crit),
                                      cmap=cm.sunburst, vmin=0, vmax=1))
    ax[i].set_aspect(1)
    ax[i].set_yticks(np.arange(n_site)+0.5, minor=False)
    ax[i].set_yticklabels(param['site_name'], fontsize=20)
    ax[i].text(24, 26.5, param['name'][i], fontsize=28, color='k', fontweight='bold',
               va='top', ha='right')

    # ax[i].spines['bottom'].set_visible(False)
    # ax[i].spines['top'].set_visible(False)
    # ax[i].spines['left'].set_visible(False)
    # ax[i].spines['right'].set_visible(False)

    if i == 2:
        ax[i].set_xticks(np.arange(-24, 30, 6))
        ax[i].tick_params(axis='x', labelsize=20)
        ax[i].set_xlabel(index_name[param['mode']] + ' lead (months)', size=24)
    else:
        ax[i].tick_params(axis='x', bottom=False, labelbottom=False)

cb1 = plt.colorbar(corr_vis1[0], cax=ax[3])
cb1.set_ticks([])

cb2 = plt.colorbar(corr_vis2[0], cax=ax[4])
cb2.set_label('Correlation coefficient', size=24)
ax[4].tick_params(axis='y', labelsize=22)

plt.savefig(fh['fig'], dpi=300, bbox_inches="tight")
