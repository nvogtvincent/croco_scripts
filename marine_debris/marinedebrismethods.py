#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods to set up a marine debris simulation using CMEMS data
@author: noam
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy import ndimage
import os.path
import numba

@numba.jit(nopython=True, parallel=True)
def id_coast(mask):
    # Extracts coast cells from a mask
    coast = np.zeros_like(mask)

    ny = np.shape(coast)[0]
    nx = np.shape(coast)[1]

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            coastcheck = mask[i-1:i+2, j-1:j+2]

            if (np.sum(coastcheck) >= 1) and (mask[i, j] == 0):
                coast[i, j] = 1

    return coast


def cmems_proc(model_fh, mask_fh):
    # This function takes output from CMEMS GLORYS12V1 and generates
    # Labelled coast cells
    # A land mask
    # Cell boundaries (boundary points)

    if os.path.isfile(mask_fh):
        with Dataset(mask_fh, mode='r') as nc:
            coast = np.array(nc.variables['coast'][:])
            lsm = np.array(nc.variables['lsm'][:])
            lon = np.array(nc.variables['longitude'][:])
            lon_bnd = np.array(nc.variables['longitude_bnd'][:])
            lat = np.array(nc.variables['latitude'][:])
            lat_bnd = np.array(nc.variables['latitude_bnd'][:])
    else:
        with Dataset(model_fh, mode='r') as nc:
            lon = np.array(nc.variables['longitude'][:])
            lat = np.array(nc.variables['latitude'][:])
            uo  = np.array(nc.variables['uo'][0, 0, :, :])

        # Generate boundary points
        dx = lon[1] - lon[0]
        dy = lat[1] - lat[0]

        lon_bnd = np.linspace(lon[0] - dx/2,
                              lon[-1] + dx/2,
                              num = len(lon) + 1)

        lat_bnd = np.linspace(lat[0] - dx/2,
                              lat[-1] + dx/2,
                              num = len(lat) + 1)

        # Generate the mask
        lsm = np.zeros_like(uo)
        lsm[uo == -32767] = 1  # Set land cells to 1

        # Identify coast cells
        coast = id_coast(lsm)  # Where coast cells are 1

        # Save to netcdf
        with Dataset(mask_fh, mode='w') as nc:
            # Create the dimensions
            nc.createDimension('lon', len(lon))
            nc.createDimension('lat', len(lat))
            nc.createDimension('lon_bnd', len(lon_bnd))
            nc.createDimension('lat_bnd', len(lat_bnd))

            nc.createVariable('longitude', 'f4', ('lon'), zlib=True)
            nc.variables['longitude'].long_name = 'longitude'
            nc.variables['longitude'].units = 'degrees_east'
            nc.variables['longitude'].standard_name = 'longitude'
            nc.variables['longitude'][:] = lon

            nc.createVariable('latitude', 'f4', ('lat'), zlib=True)
            nc.variables['latitude'].long_name = 'latitude'
            nc.variables['latitude'].units = 'degrees_north'
            nc.variables['latitude'].standard_name = 'latitude'
            nc.variables['latitude'][:] = lat

            nc.createVariable('longitude_bnd', 'f4', ('lon_bnd'), zlib=True)
            nc.variables['longitude_bnd'].long_name = 'longitude_bound'
            nc.variables['longitude_bnd'].units = 'degrees_east'
            nc.variables['longitude_bnd'].standard_name = 'longitude_bound'
            nc.variables['longitude_bnd'][:] = lon_bnd

            nc.createVariable('latitude_bnd', 'f4', ('lat_bnd'), zlib=True)
            nc.variables['latitude_bnd'].long_name = 'latitude_bound'
            nc.variables['latitude_bnd'].units = 'degrees_north'
            nc.variables['latitude_bnd'].standard_name = 'latitude_bound'
            nc.variables['latitude_bnd'][:] = lat_bnd

            nc.createVariable('lsm', 'i2', ('lat', 'lon'), zlib=True)
            nc.variables['lsm'].long_name = 'land_sea_mask'
            nc.variables['lsm'].units = '1 = Land, 0 = Sea'
            nc.variables['lsm'].standard_name = 'land_sea_mask'
            nc.variables['lsm'][:] = lsm

            nc.createVariable('coast', 'i2', ('lat', 'lon'), zlib=True)
            nc.variables['coast'].long_name = 'coast_mask'
            nc.variables['coast'].units = '1 = Coast, 0 = Not coast'
            nc.variables['coast'].standard_name = 'coast_mask'
            nc.variables['coast'][:] = coast

    return coast, lsm, lon, lat, lon_bnd, lat_bnd

def one_release(lon0, lon1, lat0, lat1, coast, coast_lon, coast_lat,
                coast_lon_bnd, coast_lat_bnd, pn):

    # This script generates the release locations based on the coastline mask
    # and a release region

    # Firstly subset the domain
    j0 = np.searchsorted(coast_lon, lon0)
    j1 = np.searchsorted(coast_lon, lon1)
    i0 = np.searchsorted(coast_lat, lat0)
    i1 = np.searchsorted(coast_lat, lat1)

    coast = coast[i0:i1, j0:j1]
    coast_lon = coast_lon[j0:j1]
    coast_lon_bnd = coast_lon_bnd[j0:j1+1]
    coast_lat = coast_lat[i0:i1]
    coast_lat_bnd = coast_lat_bnd[i0:i1+1]

    # Calculate the number of particles to be released
    ncells = np.sum(coast)
    pn2 = pn*pn
    nparticles = int(ncells*pn2)

    pos0 = np.zeros((nparticles, 2))

    # Calculate particle initial positions
    comp_counter = 0
    for i in range(len(coast_lat)):
        for j in range(len(coast_lon)):
            # Check if coast cell

            if coast[i, j] == 1:
                x0 = coast_lon_bnd[j]
                x1 = coast_lon_bnd[j+1]
                y0 = coast_lat_bnd[i]
                y1 = coast_lat_bnd[i+1]

                # Calculate the particle spacing
                dx = (x1 - x0)/(pn+1)
                dy = (y1 - y0)/(pn+1)

                # Notify user of the particle spacing
                if comp_counter == 0:
                    spacing = int(np.pi*6371000/180*dy)
                    print('Particle spacing is c. ' + str(spacing) + 'm')

                # dx/2 and dy/2 spacing is needed on the boundaries of each cell to
                # prevent overlaps between adjacent cells
                x0 += 0.5*dx
                x1 -= 0.5*dx
                y0 += 0.5*dy
                y1 -= 0.5*dy

                x_lin = np.linspace(x0, x1, num=pn)
                y_lin = np.linspace(y0, y1, num=pn)

                x, y = np.meshgrid(x_lin, y_lin)
                x = x.flatten()
                y = y.flatten()

                pos0[comp_counter*pn2:(comp_counter+1)*pn2, 0] = x
                pos0[comp_counter*pn2:(comp_counter+1)*pn2, 1] = y

                comp_counter += 1

    return pos0

def time_stagger(pos0, last_release, release_number, release_interval):
    npart = np.shape(pos0)[0]
    pos = np.zeros((npart*release_number, 3))

    for t in range(release_number):
        time = t*(release_interval*3600*24)  # convert to s
        time = (3600*24*last_release) - time              # subtract from last
        pos[t*npart:(t+1)*npart, :2] = pos0
        pos[t*npart:(t+1)*npart, 2]  = time

    return pos

