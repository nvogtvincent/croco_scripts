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

@numba.jit(nopython=True)
def psi_mask(lsm_rho, lsm_psi):
    # This function generates a land-sea mask on an empty psi grid from the
    # rho grid
    for i in range(np.shape(lsm_psi)[0]):
        for j in range(np.shape(lsm_psi)[1]):
            if j == np.shape(lsm_psi)[1]-1:
                # Wraparound (special case for last column)
                val = (lsm_rho[i, j] +
                       lsm_rho[i + 1, j] +
                       lsm_rho[i, 0] +
                       lsm_rho[i + 1, 0])
            else:
                val = (lsm_rho[i, j] +
                       lsm_rho[i+1, j] +
                       lsm_rho[i, j+1] +
                       lsm_rho[i+1, j+1])

            if val == 4:
                lsm_psi[i, j] = 1

    return lsm_psi



def cmems_proc(model_fh, mask_fh):
    # This function takes output from CMEMS GLORYS12V1 and generates
    # Labelled coast cells
    # A land mask
    # Cell boundaries (boundary points)

    if os.path.isfile(mask_fh):
        with Dataset(mask_fh, mode='r') as nc:
            coast = np.array(nc.variables['coast'][:])
            groups = np.array(nc.variables['groups'][:])
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
            fill = nc.variables['uo']._FillValue

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
        lsm[uo == fill] = 1  # Set land cells to 1

        # Identify coast cells
        coast = id_coast(lsm)  # Where coast cells are 1

        # Exclude the Mediterannean
        med_s_lat = 30
        med_s_index = np.searchsorted(lat, med_s_lat)
        med_e_lon = 42
        med_e_index = np.searchsorted(lon, med_e_lon)

        coast[med_s_index:, :med_e_index] = 0

        # Now group coast cells into 10x10 groups
        # Format:
        # ID: XXXYYY
        # XXX = group number (100-999)
        # YYY = cell number in group (000-099)

        gsize = 24
        ngroups_y = int(np.floor(len(lat)/gsize))
        ngroups_x = int(np.floor(len(lon)/gsize))  # Floor -> last lines ignored

        groups = np.zeros_like(coast)
        group_count = 0

        template = np.reshape(np.linspace(0, (gsize**2)-1,
                                          num=gsize**2,
                                          dtype=np.int32),
                              (gsize, gsize))

        for i in range(ngroups_y):
            for j in range(ngroups_x):
                # Firstly check if there are any coast cells
                subset = coast[i*gsize:(i+1)*gsize, j*gsize:(j+1)*gsize]
                if np.sum(subset) > 0:
                    subset = template + 1E5 + group_count*1E3
                    groups[i*gsize:(i+1)*gsize, j*gsize:(j+1)*gsize] = subset
                    group_count += 1

        groups[coast == 0] = 0  # Only have nonzero values for ocean points
        print(str(group_count) + ' groups in total.')

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

            nc.createVariable('groups', 'i4', ('lat', 'lon'), zlib=True,
                              fill_value=0)
            nc.variables['groups'].long_name = 'groups'
            nc.variables['groups'].units = '[XXX/YYY] XXX: group number, YYY: cell index'
            nc.variables['groups'].standard_name = 'groups'
            nc.variables['groups'][:] = groups

    return coast, groups, lsm, lon, lat, lon_bnd, lat_bnd

def cmems_globproc(model_fh, mask_fh):
    # This function takes output from global CMEMS GLORYS12V1 and generates
    # Labelled coast cells
    # A land mask
    # Cell boundaries (boundary points)

    # Note:
    # Velocities are defined on the rho grid
    # Land-sea mask/coast are defined on the psi grid

    if os.path.isfile(mask_fh):
        with Dataset(mask_fh, mode='r') as nc:
            coast_psi = np.array(nc.variables['coast_psi'][:])
            lsm_psi = np.array(nc.variables['lsm_psi'][:])
            lon_rho = np.array(nc.variables['lon_rho'][:])
            lon_psi = np.array(nc.variables['lon_psi'][:])
            lat_rho = np.array(nc.variables['lat_rho'][:])
            lat_psi = np.array(nc.variables['lat_psi'][:])
    else:
        with Dataset(model_fh, mode='r') as nc:
            lon = np.array(nc.variables['longitude'][:])
            lat = np.array(nc.variables['latitude'][:])
            lon_rho, lat_rho = np.meshgrid(lon, lat)

            uo  = np.array(nc.variables['uo'][0, 0, :, :])
            fill = nc.variables['uo']._FillValue

        # Generate the global psi grid
        lat_psi = np.linspace(-80 + (1/24), 90 - (1/24), num=170*12)
        lon_psi = np.linspace(-180 + (1/24), 180 - (1/24), num=360*12)

        lon_psi, lat_psi = np.meshgrid(lon_psi, lat_psi)

        # Now generate the mask on the lat/lon (rho) grid
        lsm_rho = np.zeros_like(uo, dtype=np.int8)
        lsm_rho[uo == fill] = 1  # Set land cells to 1

        # Now generate the ('true') mask on the lat/lon (psi) grid:
        # A node on the psi grid is 'land' if all 4 surrounding nodes on the
        # rho grid are also land
        lsm_psi = np.zeros_like(lat_psi)
        lsm_psi = psi_mask(lsm_rho, lsm_psi)

        # Identify coast cells
        coast_psi = id_coast(lsm_psi)  # Where coast cells are 1

        # Now group coast cells into 10x10 groups
        # Format:
        # ID: XXXYYY
        # XXX = group number (100-999)
        # YYY = cell number in group (000-099)

        # gsize = 24
        # ngroups_y = int(np.floor(len(lat)/gsize))
        # ngroups_x = int(np.floor(len(lon)/gsize))  # Floor -> last lines ignored

        # groups = np.zeros_like(coast)
        # group_count = 0

        # template = np.reshape(np.linspace(0, (gsize**2)-1,
        #                                   num=gsize**2,
        #                                   dtype=np.int32),
        #                       (gsize, gsize))

        # for i in range(ngroups_y):
        #     for j in range(ngroups_x):
        #         # Firstly check if there are any coast cells
        #         subset = coast[i*gsize:(i+1)*gsize, j*gsize:(j+1)*gsize]
        #         if np.sum(subset) > 0:
        #             subset = template + 1E5 + group_count*1E3
        #             groups[i*gsize:(i+1)*gsize, j*gsize:(j+1)*gsize] = subset
        #             group_count += 1

        # groups[coast == 0] = 0  # Only have nonzero values for ocean points
        # print(str(group_count) + ' groups in total.')

        # Save to netcdf
        with Dataset(mask_fh, mode='w') as nc:
            # Create the dimensions
            nc.createDimension('lon_rho', np.shape(lon_rho)[1])
            nc.createDimension('lat_rho', np.shape(lon_rho)[0])
            nc.createDimension('lon_psi', np.shape(lon_psi)[1])
            nc.createDimension('lat_psi', np.shape(lon_psi)[0])

            nc.createVariable('lon_psi', 'f4', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['lon_psi'].long_name = 'longitude_on_psi_points'
            nc.variables['lon_psi'].units = 'degrees_east'
            nc.variables['lon_psi'].standard_name = 'longitude_psi'
            nc.variables['lon_psi'][:] = lon_psi

            nc.createVariable('lat_psi', 'f4', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['lat_psi'].long_name = 'latitude_on_psi_points'
            nc.variables['lat_psi'].units = 'degrees_north'
            nc.variables['lat_psi'].standard_name = 'latitude_psi'
            nc.variables['lat_psi'][:] = lat_psi

            nc.createVariable('lon_rho', 'f4', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['lon_rho'].long_name = 'longitude_on_rho_points'
            nc.variables['lon_rho'].units = 'degrees_east'
            nc.variables['lon_rho'].standard_name = 'longitude_rho'
            nc.variables['lon_rho'][:] = lon_rho

            nc.createVariable('lat_rho', 'f4', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['lat_rho'].long_name = 'latitude_on_rho_points'
            nc.variables['lat_rho'].units = 'degrees_north'
            nc.variables['lat_rho'].standard_name = 'latitude_rho'
            nc.variables['lat_rho'][:] = lat_rho

            nc.createVariable('lsm_psi', 'i2', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['lsm_psi'].long_name = 'land_sea_mask_on_psi_points'
            nc.variables['lsm_psi'].units = '1 = Land, 0 = Sea'
            nc.variables['lsm_psi'].standard_name = 'land_sea_mask_psi'
            nc.variables['lsm_psi'][:] = lsm_psi

            nc.createVariable('lsm_rho', 'i2', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['lsm_rho'].long_name = 'land_sea_mask_on_rho_points'
            nc.variables['lsm_rho'].units = '1 = Land, 0 = Sea'
            nc.variables['lsm_rho'].standard_name = 'land_sea_mask_rho'
            nc.variables['lsm_rho'][:] = lsm_rho

            nc.createVariable('coast_psi', 'i2', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['coast_psi'].long_name = 'coast_mask_on_psi_points'
            nc.variables['coast_psi'].units = '1 = Coast, 0 = Not coast'
            nc.variables['coast_psi'].standard_name = 'coast_mask_psi'
            nc.variables['coast_psi'][:] = coast_psi

            # nc.createVariable('groups', 'i4', ('lat', 'lon'), zlib=True,
            #                   fill_value=0)
            # nc.variables['groups'].long_name = 'groups'
            # nc.variables['groups'].units = '[XXX/YYY] XXX: group number, YYY: cell index'
            # nc.variables['groups'].standard_name = 'groups'
            # nc.variables['groups'][:] = groups

    return coast_psi, lsm_psi, lon_rho, lat_rho, lon_psi, lat_psi


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

