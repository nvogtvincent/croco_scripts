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
from scipy.ndimage import distance_transform_edt

@numba.jit(nopython=True)
def psi_mask(lsm_rho, psi_grid):
    # This function generates a land-sea mask on an empty psi grid from the
    # rho grid
    # Land nodes on the psi grid are defined as being at the centre of 4 land
    # nodes on the rho grid:
    # Coast nodes on the psi grid are defined as being at the centre of 1-3
    # land nodes on the rho grid:

    # p_______p
    # |   :   |
    # |...r...|
    # |   :   |
    # p_______p

    lsm_psi   = np.zeros_like(psi_grid)
    coast_psi = np.zeros_like(psi_grid)

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
            elif val in [1, 2, 3]:
                coast_psi[i, j] = 1

    return lsm_psi, coast_psi

@numba.jit(nopython=True)
def cnorm(lsm_rho):
    # This function generates a vector field normal to the land-sea mask, at
    # coastal rho points

    cnormx = np.zeros_like(lsm_rho)
    cnormy = np.zeros_like(lsm_rho)

    for i in range(lsm_rho.shape[0]-2):
        for j in range(lsm_rho.shape[1]-2):
            if (i > 0)*(j > 0):
                # Extract 3x3 stencil
                block = lsm_rho[i-1:i+2, j-1:j+2]

                # Check if coastal rho point (is land AND has >0 adj. ocean)
                if (lsm_rho[i, j])*(np.sum(block) < 9):
                    dx = block[1, 2] - block[1, 0]
                    dy = block[2, 1] - block[0, 1]

                    # If dx or dy are nonzero, it is not a diagonal coast cell
                    if (np.abs(dx) + np.abs(dy) == 0):
                        dxy = block[2, 2] - block[0, 0]
                        dyx = block[2, 0] - block[0, 2]

                        dx = dxy - dyx
                        dy = dxy + dyx

                    cnormx[i, j] = -dx
                    cnormy[i, j] = -dy

    return cnormx, cnormy


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
            contents = {'coast_psi': np.array(nc.variables['coast_psi'][:]),
                        'lsm_psi': np.array(nc.variables['lsm_psi'][:]),
                        'lsm_rho': np.array(nc.variables['lsm_rho'][:]),
                        'lon_rho': np.array(nc.variables['lon_rho'][:]),
                        'lon_psi': np.array(nc.variables['lon_psi'][:]),
                        'lat_rho': np.array(nc.variables['lat_rho'][:]),
                        'lat_psi': np.array(nc.variables['lat_psi'][:]),
                        'cdist_rho': np.array(nc.variables['cdist_rho'][:]),
                        'cnormx_rho': np.array(nc.variables['cnormx_rho'][:]),
                        'cnormy_rho': np.array(nc.variables['cnormy_rho'][:])}

    else:
        with Dataset(model_fh, mode='r') as nc:
            lon_rho = np.array(nc.variables['longitude'][:])
            lat_rho = np.array(nc.variables['latitude'][:])

            uo  = np.array(nc.variables['uo'][0, 0, :, :])
            fill = nc.variables['uo']._FillValue

        # Generate the global psi grid
        lat_psi = np.linspace(-80 + (1/24), 90 - (1/24), num=170*12)
        lon_psi = np.linspace(-180 + (1/24), 180 - (1/24), num=360*12)

        # Now generate the mask on the lat/lon (rho) grid
        lsm_rho = np.zeros_like(uo, dtype=np.int8)
        lsm_rho[uo == fill] = 1  # Set land cells to 1

        # Now generate the ('true') mask on the lat/lon (psi) grid:
        # A node on the psi grid is 'land' if all 4 surrounding nodes on the
        # rho grid are also land
        # Also generate the coast mask on the psi grid, defined as coast if
        # 1, 2 or 3 surrounding nodes are land.
        psi_grid = np.zeros((len(lat_psi), len(lon_psi)))
        lsm_psi, coast_psi = psi_mask(lsm_rho, psi_grid)

        # Now generate a rho grid with the distance to the coast
        cdist_rho = distance_transform_edt(1-lsm_rho)

        # Now generate vectors pointing away from the coast
        cnormx, cnormy = cnorm(lsm_rho)

        # Save to netcdf
        with Dataset(mask_fh, mode='w') as nc:
            # Create the dimensions
            nc.createDimension('lon_rho', len(lon_rho))
            nc.createDimension('lat_rho', len(lat_rho))
            nc.createDimension('lon_psi', len(lon_psi))
            nc.createDimension('lat_psi', len(lat_psi))

            nc.createVariable('lon_psi', 'f4', ('lon_psi'), zlib=True)
            nc.variables['lon_psi'].long_name = 'longitude_on_psi_points'
            nc.variables['lon_psi'].units = 'degrees_east'
            nc.variables['lon_psi'].standard_name = 'longitude_psi'
            nc.variables['lon_psi'][:] = lon_psi

            nc.createVariable('lat_psi', 'f4', ('lat_psi'), zlib=True)
            nc.variables['lat_psi'].long_name = 'latitude_on_psi_points'
            nc.variables['lat_psi'].units = 'degrees_north'
            nc.variables['lat_psi'].standard_name = 'latitude_psi'
            nc.variables['lat_psi'][:] = lat_psi

            nc.createVariable('lon_rho', 'f4', ('lon_rho'), zlib=True)
            nc.variables['lon_rho'].long_name = 'longitude_on_rho_points'
            nc.variables['lon_rho'].units = 'degrees_east'
            nc.variables['lon_rho'].standard_name = 'longitude_rho'
            nc.variables['lon_rho'][:] = lon_rho

            nc.createVariable('lat_rho', 'f4', ('lat_rho'), zlib=True)
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

            nc.createVariable('cdist_rho', 'f4', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['cdist_rho'].long_name = 'distance_from_coast_rho_points'
            nc.variables['cdist_rho'].units = 'grid_cells'
            nc.variables['cdist_rho'].standard_name = 'cdist_rho'
            nc.variables['cdist_rho'][:] = cdist_rho

            nc.createVariable('cnormx_rho', 'f4', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['cnormx_rho'].long_name = 'normal_at_coast_rho_x_component'
            nc.variables['cnormx_rho'].units = 'no units'
            nc.variables['cnormx_rho'].standard_name = 'cnormx_rho'
            nc.variables['cnormx_rho'][:] = cnormx

            nc.createVariable('cnormy_rho', 'f4', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['cnormy_rho'].long_name = 'normal_at_coast_rho_y_component'
            nc.variables['cnormy_rho'].units = 'no units'
            nc.variables['cnormy_rho'].standard_name = 'cnormy_rho'
            nc.variables['cnormy_rho'][:] = cnormy

            nc.createVariable('coast_psi', 'i2', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['coast_psi'].long_name = 'coast_mask_on_psi_points'
            nc.variables['coast_psi'].units = '1 = Coast, 0 = Not coast'
            nc.variables['coast_psi'].standard_name = 'coast_mask_psi'
            nc.variables['coast_psi'][:] = coast_psi

            contents = {'coast_psi': np.array(nc.variables['coast_psi'][:]),
                        'lsm_psi': np.array(nc.variables['lsm_psi'][:]),
                        'lsm_rho': np.array(nc.variables['lsm_rho'][:]),
                        'lon_rho': np.array(nc.variables['lon_rho'][:]),
                        'lon_psi': np.array(nc.variables['lon_psi'][:]),
                        'lat_rho': np.array(nc.variables['lat_rho'][:]),
                        'lat_psi': np.array(nc.variables['lat_psi'][:]),
                        'cdist_rho': np.array(nc.variables['cdist_rho'][:]),
                        'cnormx_rho': np.array(nc.variables['cnormx_rho'][:]),
                        'cnormy_rho': np.array(nc.variables['cnormy_rho'][:])}

    return contents


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

