#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:29:47 2021
This script adds the ubar/vbar variables to the CMEMS GLORYS12V1 download for
use in the CROCO model
@author: noam
"""
from netCDF4 import Dataset
import numpy as np
import sys
import os
from progress.bar import Bar

##############################################################################
# File locations #############################################################
##############################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))
year = input('Which year?')
fh = this_dir + '/CMEMS_' + str(year) + '.nc'

##############################################################################
# Modify the file ############################################################
##############################################################################

with Dataset(fh, mode='a') as fh:
    u = np.array(fh.variables['uo'][:])
    v = np.array(fh.variables['vo'][:])
    z = np.array(fh.variables['depth'][:])
    time_ = np.array(fh.variables['time'][:])
    dz = np.gradient(z)
    missing_data = -32767.0

    # Generate the barotropic velocities
    nTime = len(time_)
    ubar = np.zeros((nTime,np.shape(u)[2],np.shape(u)[3]))
    vbar = np.zeros((nTime,np.shape(u)[2],np.shape(u)[3]))

    # Create the new ubar/vbar variables
    fh_ubar = fh.createVariable('ubar', 'f8', ('time', 'latitude', 'longitude'))
    fh_ubar.setncatts({'_FillValue':missing_data,\
            'long_name': u'Eastward barotropic velocity',\
            'standard_name': u'eastward_barotropic_sea_water_velocity',\
            'units': u'm s-1',\
            'unit_long': u'Meters per second',\
            'cell_methods': u'area: mean',\
            'add_offset': 0.0,\
            'scale_factor': 1.0})

    fh_vbar = fh.createVariable('vbar', 'f8', ('time', 'latitude', 'longitude'))
    fh_vbar.setncatts({'_FillValue':missing_data,\
            'long_name': u'Northward barotropic velocity',\
            'standard_name': u'northward_barotropic_sea_water_velocity',\
            'units': u'm s-1',\
            'unit_long': u'Meters per second',\
            'cell_methods': u'area: mean',\
            'add_offset': 0.0,\
            'scale_factor': 1.0})

    # Loop through the months
    for day in range(nTime):
        bar = Bar('Calculating ubar and vbar...', max=nTime)
        # load u, v and depth
        for xi in range(np.shape(u)[3]):
            for yi in range(np.shape(u)[2]):
                if u[day,0,yi,xi] != missing_data:
                    # Firstly calculate the depth
                    z_mask = np.copy(u[day,:,yi,xi])
                    z_mask[z_mask == missing_data] = 0
                    z_mask[z_mask != 0] = 1
                    ubar[day,yi,xi] = np.sum(u[day,:,yi,xi]*z_mask*dz)/np.sum(dz*z_mask)
                    vbar[day,yi,xi] = np.sum(v[day,:,yi,xi]*z_mask*dz)/np.sum(dz*z_mask)
                else:
                    ubar[day,yi,xi] = missing_data
                    vbar[day,yi,xi] = missing_data
        bar.next()
    bar.finish()

    fh['ubar'][:,:,:] = ubar
    fh['vbar'][:,:,:] = vbar

    print('Processing complete!')