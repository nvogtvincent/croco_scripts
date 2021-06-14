#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate a moving average from CROCO output data for very large
datasets
@author: noam
"""

from netCDF4 import Dataset
from scipy.ndimage import convolve1d
import numpy as np
import sys
import os
from progress.bar import Bar

##############################################################################
# Parameters #################################################################
##############################################################################

out_freq = 3    # Hours
window   = 24   # Hours

in_fh = '1993_vis.nc'
out_fh = '1993_vis_filtered.nc'

variables = ['temp_surf']

##############################################################################
# File locations #############################################################
##############################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))
in_fh = this_dir + '/' + in_fh
out_fh = this_dir + '/' + out_fh

##############################################################################
# Generate output file #######################################################
##############################################################################

# Firstly load to obtain dimensions
with Dataset(in_fh, mode='r') as nc:
    print('Loading files...')
    time = np.shape(nc.variables['time_counter_bounds'][:, 0])[0]
    lon = nc.variables['lon_rho'][:]
    lat = nc.variables['lat_rho'][:]

# Now generate the output file
with Dataset(out_fh, mode='w') as nc:
    print('Generating output netcdf...')
    # Create the dimensions
    nc.createDimension('time', time)
    nc.createDimension('lon', np.shape(lon)[1])
    nc.createDimension('lat', np.shape(lon)[0])

    # Create the time axis
    nc.createVariable('time', 'f8', ('time'), zlib=True)
    nc.variables['time'].long_name = 'seconds_since_start'
    nc.variables['time'].units = 'seconds'
    nc.variables['time'].field = 'time, scalar, series'
    nc.variables['time'].standard_name = 'seconds_since_start'
    nc.variables['time'][:] = time

    nc.createVariable('longitude', 'f8', ('lat', 'lon'), zlib=True)
    nc.variables['longitude'].long_name = 'longitude'
    nc.variables['longitude'].units = 'degrees_east'
    nc.variables['longitude'].field = 'longitude, scalar, series'
    nc.variables['longitude'].standard_name = 'longitude'
    nc.variables['longitude'][:] = lon

    nc.createVariable('latitude', 'f8', ('lat', 'lon'), zlib=True)
    nc.variables['latitude'].long_name = 'latitude'
    nc.variables['latitude'].units = 'degrees_north'
    nc.variables['latitude'].field = 'latitude, scalar, series'
    nc.variables['latitude'].standard_name = 'latitude'
    nc.variables['latitude'][:] = lat

    # Create the other variables
    if 'temp_surf' in variables:
        nc.createVariable('sst', 'f8', ('time', 'lat', 'lon'))
        nc.variables['sst'].long_name = 'filtered_sea_surface_temperature'
        nc.variables['sst'].units = 'Celsius'
        nc.variables['sst'].field = 'temperature, scalar, series'
        nc.variables['sst'].standard_name = 'sst'
        nc.variables['sst'].coordinates = 'lat lon'
    if 'salt_surf' in variables:
        nc.createVariable('sss', 'f8', ('time', 'lat', 'lon'))
        nc.variables['sss'].long_name = 'filtered_sea_surface_salinity'
        nc.variables['sss'].units = 'PSU'
        nc.variables['sss'].field = 'salinity, scalar, series'
        nc.variables['sss'].standard_name = 'sss'
        nc.variables['sss'].coordinates = 'lat lon'
    if 'zeta' in variables:
        nc.createVariable('ssh', 'f8', ('time', 'lat', 'lon'))
        nc.variables['ssh'].long_name = 'filtered_sea_surface_height'
        nc.variables['ssh'].units = 'meter'
        nc.variables['ssh'].field = 'free-surface, scalar, series'
        nc.variables['ssh'].standard_name = 'ssh'
        nc.variables['ssh'].coordinates = 'lat lon'

# Set up the filter
window_len = int(window/out_freq)
croco_filter = (1/window_len)*np.ones(window_len)

# Then loop through the columns
bar = Bar('Filtering data...', max=np.shape(lon)[0])
for i in range(np.shape(lon)[0]):
    with Dataset(in_fh, mode='r') as nc:
        if 'temp_surf' in variables:
            temp = nc.variables['temp_surf'][:, i, :]
            temp = convolve1d(temp, croco_filter, axis=0)
        if 'salt_surf' in variables:
            salt = nc.variables['salt_surf'][:, i, :]
            salt = convolve1d(salt, croco_filter, axis=0)
        if 'zeta' in variables:
            zeta = nc.variables['zeta'][:, i, :]
            zeta = convolve1d(zeta, croco_filter, axis=0)

    with Dataset(out_fh, mode='r+') as nc:
        if 'temp_surf' in variables:
            nc.variables['sst'][:, i, :] = temp
        if 'salt_surf' in variables:
            nc.variables['sss'][:, i, :] = salt
        if 'zeta' in variables:
            nc.variables['ssh'][:, i, :] = zeta

    bar.next()
bar.finish()

print('Processing complete!')


