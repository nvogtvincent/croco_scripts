#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate climatological fields from CROCO multi-year sfc output
@author: noam
"""

from netCDF4 import Dataset
import numpy as np
import sys
import os

##############################################################################
# Parameters #################################################################
##############################################################################

climvar = 'v'

##############################################################################
# File locations #############################################################
##############################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))

# Generator for file names
namelist = []
for i in range(2, 10):
    namelist.append('Y' + str(i+1) + '_30M.nc')

n_files = len(namelist)

# Check for climvar
if climvar in ['temp', 'salt', 'u', 'v']:
    climvar = climvar + '_surf'
elif climvar != 'zeta':
    raise ValueError('Unknown variable')


# List of month lengths
day_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
day_cumsum = np.cumsum(day_per_month)

if climvar in ['u_surf', 'v_surf']:
    files_per_day = 48
else:
    files_per_day = 1

# Output
output_file = this_dir + '/' + 'croco_climatology_' + climvar + '.nc'

##############################################################################
# Calculate climatology ######################################################
##############################################################################

for i in range(n_files):
    input_file = this_dir + '/' + namelist[i]
    print(input_file)

    with Dataset(input_file, mode='r') as nc:
        if i == 0:
            # Set up the output array if this is the first file
            if climvar in ['temp_surf', 'salt_surf', 'zeta']:
                lon = nc.variables['lon_rho'][:]
                lat = nc.variables['lat_rho'][:]
            elif climvar in ['u_surf']:
                lon = nc.variables['nav_lon_u'][:]
                lat = nc.variables['nav_lat_u'][:]
            else:
                lon = nc.variables['nav_lon_v'][:]
                lat = nc.variables['nav_lat_v'][:]

            climatology = np.zeros((12, np.shape(lon)[0], np.shape(lon)[1]))

        for t in range(365*files_per_day):
            month = np.searchsorted(day_cumsum, t/files_per_day, side='right')
            climatology[month, :, :] += (
                (1/(files_per_day*day_per_month[month]*n_files)) *
                nc.variables[climvar][t, :, :])


# Create the climatology to a new file
with Dataset(output_file, mode='w') as nc:
    print('Writing fields to output netcdf...')
    # Create the dimensions
    nc.createDimension('time', 12)
    nc.createDimension('lon', np.shape(lon)[1])
    nc.createDimension('lat', np.shape(lon)[0])

    # Create the time axis
    nc.createVariable('time', 'f8', ('time'), zlib=True)
    nc.variables['time'].long_name = 'time'
    nc.variables['time'].units = 'months'
    nc.variables['time'].field = 'time, scalar, series'
    nc.variables['time'].standard_name = 'climatology_time'
    nc.variables['time'].axis = 'T'
    nc.variables['time'][:] = np.linspace(1, 12, num=12, dtype=np.int8)

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
    if climvar == 'temp_surf':
        nc.createVariable('sst', 'f8', ('time', 'lat', 'lon'))
        nc.variables['sst'].long_name = 'sea_surface_temperature'
        nc.variables['sst'].units = 'Celsius'
        nc.variables['sst'].field = 'temperature, scalar, series'
        nc.variables['sst'].standard_name = 'sst'
        nc.variables['sst'].coordinates = 'lat lon'
        nc.variables['sst'][:] = climatology
    elif climvar == 'salt_surf':
        nc.createVariable('sss', 'f8', ('time', 'lat', 'lon'))
        nc.variables['sss'].long_name = 'sea_surface_salinity'
        nc.variables['sss'].units = 'PSU'
        nc.variables['sss'].field = 'salinity, scalar, series'
        nc.variables['sss'].standard_name = 'sss'
        nc.variables['sss'].coordinates = 'lat lon'
        nc.variables['sss'][:] = climatology
    elif climvar == 'zeta':
        nc.createVariable('ssh', 'f8', ('time', 'lat', 'lon'))
        nc.variables['ssh'].long_name = 'sea_surface_height'
        nc.variables['ssh'].units = 'meter'
        nc.variables['ssh'].field = 'free-surface, scalar, series'
        nc.variables['ssh'].standard_name = 'ssh'
        nc.variables['ssh'].coordinates = 'lat lon'
        nc.variables['ssh'][:] = climatology
    elif climvar == 'u_surf':
        nc.createVariable('u', 'f8', ('time', 'lat', 'lon'))
        nc.variables['u'].long_name = 'u-momentum component'
        nc.variables['u'].units = 'meter second-1'
        nc.variables['u'].field = 'u-velocity, scalar, series'
        nc.variables['u'].standard_name = 'sea_water_x_velocity_at_u_location'
        nc.variables['u'].coordinates = 'lat lon'
        nc.variables['u'][:] = climatology
    elif climvar == 'v_surf':
        nc.createVariable('v', 'f8', ('time', 'lat', 'lon'))
        nc.variables['v'].long_name = 'v-momentum component'
        nc.variables['v'].units = 'meter second-1'
        nc.variables['v'].field = 'v-velocity, scalar, series'
        nc.variables['v'].standard_name = 'sea_water_y_velocity_at_v_location'
        nc.variables['v'].coordinates = 'lat lon'
        nc.variables['v'][:] = climatology

    print('Completed!')
