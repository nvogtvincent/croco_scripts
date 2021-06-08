#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script animates SST from a year of CMEMS data
Noam Vogt-Vincent 2021
"""

from netCDF4 import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import matplotlib.animation as ani
import cmocean.cm as cmo
import cartopy.crs as ccrs


##############################################################################
# EXPLANATION ################################################################
##############################################################################

# This script takes SST and sea-ice thickness data downloaded from CMEMS' 1/12
# degree reanalysis and plots it on a rotating globe, although it can easily
# be adapted to another global gridded product. The 3D effect is obtained by
# plotting SST as a heightmap with matplotlib's light source shader.

# This script is not very efficient, but I haven't worked out a more efficient
# way of maintaining the cartopy projection whilst rotating the angle without
# redrawing frames!

##############################################################################
# File locations #############################################################
##############################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))

# Name of the files used for SST and ice data
tracer_source = this_dir + '/cmems/'

output_name = 'timelapse_cmems_2019.mp4'

ice_fh = tracer_source + 'si.nc'
sst_fh = tracer_source + 'sst.nc'

##############################################################################
# Parameters #################################################################
##############################################################################

# variable names
ice_variable_name = 'sithick'
sst_variable_name = 'thetao'

# Minimum and maximum sea ice thickness
imin = -0.5
imax = 5
icmap = cmo.ice_r

# Minimum and maximum SST
tmin = -3
tmax = 36
tcmap = cmo.thermal

ve = 20 # Vertical exaggeration

fps = 15
total_frames = 365

# List used to display the date

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

# Set up the figure and lighting
f = plt.figure(figsize=(10, 10))

# Projection parameters (initial)
proj = {
        'central_longitude': 0.0,
        'central_latitude': 0.0,
        }

ax = plt.axes(projection=ccrs.Orthographic(**proj))

img_extent = (-180., 179.91667, -80., 90.)

csfont = {'fontname': 'Ubuntu',
          'color' : "white",
          'size': 20}

f.set_facecolor("black")

f.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=None, hspace=None)

ls = LightSource(azdeg=315, altdeg=30)

# Load in the first file to set up the figure

with Dataset(sst_fh, mode='r') as nc:

    sst = np.flipud(nc.variables[sst_variable_name][0, 0, :, :])
    sst = np.ma.masked_values(sst, 0)

with Dataset(ice_fh, mode='r') as nc:

    ice = np.flipud(nc.variables[ice_variable_name][0, :, :])
    ice = np.ma.masked_values(ice, 0)

# Apply continents as a mask
continent_mask = np.ma.getmask(sst).astype(int)
continent_mask = np.stack((continent_mask,
                           continent_mask,
                           continent_mask,
                           continent_mask),
                          axis = -1)

ax.set_axis_off()

##############################################################################
# Set up animation ###########################################################
##############################################################################

def animate_ocean(t):

    # Rotate the projection
    angle = (360/total_frames)*t

    proj = {
        'central_longitude': angle,
        'central_latitude': 0.0,
        }

    ax = plt.axes(projection=ccrs.Orthographic(**proj))

    # Plot the new data
    with Dataset(sst_fh, mode='r') as nc:
        sst = np.flipud(nc.variables[sst_variable_name][t, 0, :, :])
        sst = np.ma.masked_values(sst, 0)
        sst_im = ls.shade(sst, cmap=tcmap, blend_mode='soft',
                          vert_exag=ve, vmin=tmin, vmax=tmax)
        sst_im = np.ma.masked_where(continent_mask == 0,
                            sst_im)
        im1 = ax.imshow(sst_im,
                origin='upper',
                extent=img_extent,
                transform=ccrs.PlateCarree())

    with Dataset(ice_fh, mode='r') as nc:
        ice = np.flipud(nc.variables[ice_variable_name][t, :, :])
        ice = np.ma.masked_values(ice, 0)
        ice_im = ls.shade(ice, cmap=icmap, blend_mode='soft',
                          vert_exag=120, vmin=tmin, vmax=tmax)
        ice_im = np.ma.masked_where(continent_mask == 0,
                            ice_im)
        im2 = ax.imshow(ice_im,
                origin='upper',
                extent=img_extent,
                transform=ccrs.PlateCarree())

    # Work out the current month
    month = np.searchsorted(day_cumsum, t/files_per_day, side='right')
    if month == 0:
        day = t + 1
    else:
        day = t - day_cumsum[month - 1] + 1

    month = months[month]
    day = str(day)

    f.set_facecolor('black')
    ax.set_facecolor("black")

    date_string = month + ' ' + day
    date = ax.text(0.02, 0.96, date_string, transform=ax.transAxes, **csfont)

    return [im1, im2, date]


animation = ani.FuncAnimation(f,
                              animate_ocean,
                              frames=total_frames,
                              interval=1000/fps)

animation.save(output_name,
               fps=fps,
               bitrate=16000,)
