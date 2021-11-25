#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script analyses potential connectivity between Aldabra and Seychelles Plateau
@author: Noam
"""

import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as ani
import cmasher as cmr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset

##############################################################################
# Parameters #################################################################
##############################################################################

# PARAMETERS
param = {'pnum_per_release': 0.1,},    # Particle releases per month

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)) + '/../',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/../TRAJ/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'traj_ald': dirs['traj'] + 'Ald_fwd.nc',
      'traj_sey': dirs['traj'] + 'Sey_fwd.nc',
      'fig_fh': dirs['fig'] + 'aldabra_seychelles_connectivity.png'}

##############################################################################
# Load data ##################################################################
##############################################################################

data = []

with Dataset(fh['traj_ald'], mode='r') as nc:
    data_ald = nc.variables['rt'][:]/(3600*24)

with Dataset(fh['traj_sey'], mode='r') as nc:
    data_sey = nc.variables['rt'][:]/(3600*24)

time = np.arange(1993+1/24, 2019+1/24, 1/12)+1/12
nt = len(time)
ny = nt/12
ppr_ald = int(data_ald.shape[0]/nt)
ppr_sey = int(data_sey.shape[0]/nt)

# data_ald[0] = 62.5*3600*24

# Calculate the number of particles that reach destination for each release month
# [60, 70, 80, 90, 100, 110, 120]
t_thresh = [60, 70, 80, 90, 100, 110, 120]
t_len = len(t_thresh)

# IMPORTANT: EXPERIMENT STARTED IN FEBRUARY SO SHIFT BACK BY 1!

arr_ald = np.zeros((nt, len(t_thresh)))
arr_sey = np.zeros((nt, len(t_thresh)))

for i in range(nt):
    subarr_ald = data_ald[i*ppr_ald:(i+1)*ppr_ald]
    subarr_sey = data_sey[i*ppr_sey:(i+1)*ppr_sey]

    # Remove zeros
    subarr_ald = subarr_ald[subarr_ald > 0]
    subarr_sey = subarr_sey[subarr_sey > 0]

    for j in range(t_len):
        subarr2_ald = subarr_ald[subarr_ald <= t_thresh[j]]
        subarr2_sey = subarr_sey[subarr_sey <= t_thresh[j]]

        ald_succ = len(subarr2_ald)/ppr_ald
        sey_succ = len(subarr2_sey)/ppr_sey

        arr_ald[i, j] = ald_succ
        arr_sey[i, j] = sey_succ

# Move February to pos 2 (note that this means 2019_jan is now pos1)
arr_ald = np.roll(arr_ald, 1, axis=0)
arr_sey = np.roll(arr_sey, 1, axis=0)

# Calculate monthly climatology
ald_monclim = np.zeros((t_len, 12))
sey_monclim = np.zeros((t_len, 12))

# Calculate proportion of months with any connections
ald_monconn = np.zeros((t_len, 12))
sey_monconn = np.zeros((t_len, 12))

for i in range(t_len):
    ald_monclim[i, :] = np.mean(arr_ald[:, i].reshape((-1, 12)), axis=0)
    sey_monclim[i, :] = np.mean(arr_sey[:, i].reshape((-1, 12)), axis=0)

    ald_monconn[i, :] = np.count_nonzero(arr_ald[:, i].reshape((-1, 12)), axis=0)/ny
    sey_monconn[i, :] = np.count_nonzero(arr_sey[:, i].reshape((-1, 12)), axis=0)/ny

# # Plot raw results
f = plt.figure(constrained_layout=True, figsize=(20,10))

heights = [3, 1]
spec = f.add_gridspec(ncols=2, nrows=2, height_ratios=heights)

# ALDABRA
ax1 = f.add_subplot(spec[0, 0])
clist = []
for i in range(t_len):
    cn = cmr.neutral_r.N
    color = cmr.neutral_r.colors[int(300+i*(cn*0.8/6))]
    clist.append(ax1.bar(np.arange(1,13), np.flipud(ald_monclim)[i, :], color=color, label=t_thresh[i]))

ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticks(np.arange(1, 13))
ax1.tick_params(length=0)

ax2 = f.add_subplot(spec[1, 0], sharex=ax1)
for i in range(t_len):
    cn = cmr.neutral_r.N
    color = cmr.neutral_r.colors[int(300+i*(cn*0.8/6))]
    ax2.bar(np.arange(1,13), np.flipud(ald_monconn)[i, :], color=color)

ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.get_xaxis().set_visible(False)

# SEYCHELLES
ax3 = f.add_subplot(spec[0, 1], sharey=ax1)
for i in range(t_len):
    cn = cmr.neutral_r.N
    color = cmr.neutral_r.colors[int(300+i*(cn*0.8/6))]
    ax3.bar(np.arange(1,13), np.flipud(sey_monclim)[i, :], color=color)

ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.tick_params(length=0)
ax3.set_xticks(np.arange(1, 13))
ax3.yaxis.tick_right()

ax4 = f.add_subplot(spec[1, 1], sharex=ax3)
for i in range(t_len):
    cn = cmr.neutral_r.N
    color = cmr.neutral_r.colors[int(300+i*(cn*0.8/6))]
    ax4.bar(np.arange(1,13), np.flipud(sey_monconn)[i, :], color=color)

ax4.invert_yaxis()
ax4.spines['top'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.get_xaxis().set_visible(False)
ax4.yaxis.tick_right()

ax3.legend(clist, ['120', '110', '100', '90', '80', '70', '60'],
           frameon=False)

plt.savefig(fh['fig_fh'], dpi=300)

# f.subplots_adjust(hspace=0.05)

# ax1.pcolormesh(np.arange(1850, 2016), time_axis, hist, cmap=cmr.freeze_r,
#                vmin=0, vmax=np.max(hist)*2)

# ax1.plot(decade_mp[:16], ssp245_decmean[:16], 'w-', linewidth=2, marker='s')
# # ax1.scatter(decade_mp[:17], ssp245_decmean[:17], )

# ax1.spines['top'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['left'].set_visible(False)
# ax1.get_xaxis().set_visible(False)
# ax1.set_xlim([1847, 2015])
# ax1.set_yticks(np.arange(20, 50, 5))


# ax2 = f.add_subplot(spec[1, 0])
# ax2.pcolormesh(np.arange(1850, 2016), time_axis, hist, cmap=cmr.freeze_r,
#                vmin=0, vmax=np.max(hist)*2)
# ax2.spines['top'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.set_xlim([1847, 2015])
# ax2.set_xticks(np.arange(1850, 2020, 10))
# ax2.set_yticks(np.arange(20, 50, 5))
# ax2.plot(decade_mp[:16], ssp245_decmean[:16], 'w-', linewidth=2, marker='s')


