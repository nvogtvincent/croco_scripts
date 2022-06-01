#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to assess how annual mean flux/%dest/biggest_dest changes as a function
of ub and us

@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmasher as cmr
import xarray as xr
from scipy.interpolate import interp1d
from glob import glob
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from sys import argv

# PARAMETERS
param = {# Physics
         'mode': argv[1],

         # Destination
         'destination': argv[2],

         # Source
         'source': argv[3],

         # CMAP
         'cmap': cmr.guppy_r}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/../PLASTIC_DATA/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'flux': sorted(glob(dirs['script'] + '/terrestrial_flux*.nc')),
      'source_list': dirs['plastic'] + 'country_list.in',
      'sink_list': dirs['plastic'] + 'sink_list.in',
      'fig': dirs['fig'] + str(param['mode']) + '_' + param['source'] + '_' + param['destination'] + '_sensitivity.png'}

##############################################################################
# CREATE MATRICES                                                            #
##############################################################################

ub_list = np.arange(5, 65, 5)
ub_bnds = np.append(ub_list, ub_list[-1]+1)-0.5

us_list = np.logspace(1, 3.5, 12)
us_bnds = np.append(us_list, us_list[-1]+1)-0.5

flux_matrix = np.zeros((len(ub_list), len(us_list)))
pct_src_matrix = np.zeros_like(flux_matrix)
flux_src_matrix = np.zeros_like(flux_matrix)
mode_src_matrix = np.zeros_like(flux_matrix)

##############################################################################
# EXTRACT DATA                                                               #
##############################################################################

for fh_i in tqdm(fh['flux'], total=len(fh['flux'])):
    fmatrix = xr.open_dataarray(fh_i)
    us_i = fmatrix.us # Get sinking rate
    ub_i = fmatrix.ub # Get beaching rate

    # Extract mean annual flux
    # Firstly, only consider debris arriving between 1999-01 to 2014-12 (to allow a
    # 6-year spin-up), from all sources and source times, arriving at Aldabra
    fmatrix_sub = fmatrix[:, :, :, 72:264].sum(dim=('source_time', 'source')).loc[param['destination']]

    # Find annual accumulating flux
    annual_flux = fmatrix_sub.sum()/len(np.unique(fmatrix_sub.sink_time.dt.year))
    annual_flux /= 1000 # Convert to tonnes

    # Now find annual flux from source chosen
    fmatrix_sub1 = fmatrix[:, :, :, 72:264].sum(dim=('source_time', 'sink_time')).loc[param['source'], param['destination']]
    pct_source_flux = fmatrix_sub1.sum()/len(np.unique(fmatrix_sub.sink_time.dt.year))/(1000*annual_flux)
    source_flux = fmatrix_sub1.sum()/len(np.unique(fmatrix_sub.sink_time.dt.year))
    source_flux /= 1000 # Convert to tonnes

    # Now find largest source
    src_max_i = np.argmax(fmatrix[:, :, :, 72:264].sum(dim=('source_time', 'sink_time')).loc[:, param['destination']].values)

    # Now insert into matrix
    matrix_i = np.searchsorted(ub_bnds, ub_i)-1
    matrix_j = np.searchsorted(us_bnds, us_i)-1

    flux_matrix[matrix_i, matrix_j] = annual_flux
    pct_src_matrix[matrix_i, matrix_j] = pct_source_flux
    mode_src_matrix[matrix_i, matrix_j] = src_max_i
    flux_src_matrix[matrix_i, matrix_j] = source_flux

##############################################################################
# PLOT DATA                                                                  #
##############################################################################

f = plt.figure(figsize=(30, 12), constrained_layout=True)
gs = GridSpec(2, 3, figure=f, height_ratios=[1, 0.05], hspace=0.05)
ax = []
cax = []
pcm = []
cbar = []

matrix_list = [flux_matrix, pct_src_matrix, flux_src_matrix]
cmap_list = [cmr.ember, cmr.cosmic, cmr.cosmic]
title_list = ['Annual accumulation rate',
              'Fraction of debris from ' + param['source'],
              'Annual accumulation rate from ' + param['source']]
cbar_list = ['Accumulation rate (tonnes/yr)',
             'Fraction of debris',
             'Accumulation rate (tonnes/yr)']

y_axis = np.append(ub_list, ub_list[-1]+5)-2.5

x_mp_log = np.linspace(1, 3.5, 12)
x_mp_dx = x_mp_log[1]-x_mp_log[0]
x_axis = np.append(x_mp_log, x_mp_log[-1]+x_mp_dx)-(x_mp_dx/2)

for i in range(3):
    ax.append(f.add_subplot(gs[0, i]))

    if i == 0:
        pcm.append(ax[i].pcolormesh(x_axis, y_axis, matrix_list[i], cmap=cmap_list[i],
                   norm=colors.LogNorm(vmin=np.min(matrix_list[i][np.nonzero(matrix_list[i])]), vmax=np.max(matrix_list[i]))))
    elif i == 1:
        pcm.append(ax[i].pcolormesh(x_axis, y_axis, matrix_list[i], cmap=cmap_list[i],
                                    vmin=np.min(matrix_list[i]), vmax=np.max(matrix_list[i])))
    elif i == 2:
        pcm.append(ax[i].pcolormesh(x_axis, y_axis, matrix_list[i], cmap=cmap_list[i],
                   norm=colors.LogNorm(vmin=np.min(matrix_list[i][np.nonzero(matrix_list[i])]), vmax=np.max(matrix_list[i]))))

    ax[i].set_title(title_list[i], fontdict={'fontsize': 24})

    ax[i].tick_params(axis='x', labelsize=20)
    ax[i].tick_params(axis='y', labelsize=20)
    ax[i].set_xlabel('log(1/μs) (days)', fontsize=24)

    # if i == 1:
    #     ax[i].get_yaxis().set_ticks([])
    # else:
    ax[i].set_ylabel('1/μb (days)', fontsize=24)

for i in range(3):
    cax.append(f.add_subplot(gs[1, i]))
    cbar.append(plt.colorbar(pcm[i], orientation='horizontal', cax=cax[i]))
    cax[i].tick_params(axis='x', labelsize=24)
    cax[i].set_xlabel(cbar_list[i], fontsize=20)


plt.savefig(fh['fig'], dpi=300)
