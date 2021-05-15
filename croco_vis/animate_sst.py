#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script animates SST from croco output using the matplotlib shading module
Noam Vogt-Vincent 2021
"""

from netCDF4 import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import matplotlib.animation as ani
import cmocean.cm as cmo


##############################################################################
# File locations #############################################################
##############################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))

tracer_source = this_dir + '/' + 'climato_Y8_sst.nc'
output_name = 'sst_preproc.mp4'

##############################################################################
# Parameters #################################################################
##############################################################################

tmin = 22.0
tmax = 30.0

fps = 15
total_frames = 365

day_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
months = ['January',
          'February',
          'March',
          'April',
          'May',
          'June',
          'July',
          'August',
          'September',
          'October',
          'November',
          'December']
day_cumsum = np.cumsum(day_per_month)
files_per_day = 1

##############################################################################
# Set up plot ################################################################
##############################################################################

f, ax = plt.subplots(1, 1, figsize=(20, 10))

csfont = {'fontname': 'Ubuntu'}
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

ls = LightSource(azdeg=315, altdeg=30)

with Dataset(tracer_source, mode='r') as nc:
    sst = np.flipud(nc.variables['temp_surf'][0, :, :])
    sst = np.ma.masked_values(sst, 0)
    sst_im = ls.shade(sst, cmap=cmo.balance, blend_mode='soft', vert_exag=75,
                      vmin=tmin, vmax=tmax)


im = ax.imshow(sst_im, aspect='auto')
ax.set_axis_off()

date = ax.text(30, 50, 'January 10', fontsize='20', **csfont)


##############################################################################
# Set up animation ###########################################################
##############################################################################


def animate_ocean(t):
    with Dataset(tracer_source, mode='r') as nc:
        sst = np.flipud(nc.variables['temp_surf'][t, :, :])
        sst = np.ma.masked_values(sst, 0)
        sst_im = ls.shade(sst, cmap=cmo.balance, blend_mode='soft',
                          vert_exag=75, vmin=tmin, vmax=tmax)
        im.set_array(sst_im)

        # Work out the current month
        month = np.searchsorted(day_cumsum, t/files_per_day, side='right')
        if month == 0:
            day = t + 1
        else:
            day = t - day_cumsum[month - 1] + 1

        month = months[month]
        day = str(day)

        date_string = month + ' ' + day
        date.set_text(date_string)

        return [im, date]


animation = ani.FuncAnimation(f,
                              animate_ocean,
                              frames=total_frames,
                              interval=1000/fps)

animation.save(output_name,
               fps=fps,
               bitrate=16000,)
