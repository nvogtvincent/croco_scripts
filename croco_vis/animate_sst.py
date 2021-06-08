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
# ADAPTING THIS SCRIPT TO YOUR DATA ##########################################
##############################################################################

# This is a very simple script that takes a set of SST fields (or some other
# tracer of interest) and plots it as a coloured heightmap. The script
# by default generates a 1-frame-long animation from the sample file provided
# (croco_frame.nc) but this can be adapted easily to your own data by
# modifying the file handle and selecting the correct variable name (line 42).
#
# You will also need to change the number of frames depending on the timespan
# you want the animation to cover, and how many time frames your data contains.
# You may also wish to change the SST min/max to reflect the region you are
# plotting.

##############################################################################
# File locations #############################################################
##############################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))

# Name of the file used for SST (or other) data
tracer_source = this_dir + '/' + 'croco_frame.nc'

output_name = 'sample_animation.mp4'

##############################################################################
# Parameters #################################################################
##############################################################################

# SST variable name
sst_variable_name = 'temp_surf'

# Minimum and maximum SST
tmin = 22.5
tmax = 29.5

fps = 15
total_frames = 1

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
    # Check that there is enough data in the file
    if np.shape(nc.variables[sst_variable_name][:])[0] < total_frames:
        raise ValueError('total_frames exceeds number of frames in data!')

    sst = np.flipud(nc.variables[sst_variable_name][0, :, :])

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
        sst = np.flipud(nc.variables[sst_variable_name][t, :, :])
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




im = ax.imshow(sst_im, aspect='auto')
ax.set_axis_off()

date = ax.text(30, 50, 'January 10', fontsize='20', **csfont)
