#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script grids plastic source fluxes onto the coastal grid
@Author: Noam Vogt-Vincent
"""

import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
import os.path
import gdal
import numba


##############################################################################
# Functions ##################################################################
##############################################################################

@numba.jit(nopython=True)
def haversine_np(lon1, lat1, lon2, lat2):
    # From https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

@numba.jit(nopython=True)
def cdist_calc(cnearest, lon, lat, efs):
    # Firstly evaluate target longitude and latitude
    lon_target = np.zeros_like(lon)
    lat_target = np.zeros_like(lat)

    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            target_i = cnearest[0, i, j]
            target_j = cnearest[1, i, j]

            lon_target[i, j] = lon[target_i, target_j]
            lat_target[i, j] = lat[target_i, target_j]

    # Now calculate distances between (lon, lat) and (lon_target, lat_target)
    cdist = haversine_np(lon, lat, lon_target, lat_target)

    return cdist

@numba.jit(nopython=True)
def cplastic_calc(plastic_data, cnearest):
    cplastic = np.zeros_like(plastic_data)

    for i in range(plastic_data.shape[0]):
        for j in range(plastic_data.shape[1]):
            target_i = cnearest[0, i, j]
            target_j = cnearest[1, i, j]

            cplastic[target_i, target_j] += plastic_data[i, j]

    return cplastic

@numba.jit(nopython=True)
def id_coast_cells(coast, country_id, country_id_nearest):
    coast_id = np.copy(coast)

    for i in range(coast_id.shape[0]):
        for j in range(coast_id.shape[1]):
            if coast[i, j] == 1:
                # Firstly check if the coast already has a country ID
                id0 = country_id[i, j]

                if id0 == -32768:
                    target_i = country_id_nearest[0, i, j]
                    target_j = country_id_nearest[1, i, j]
                    id0 = country_id[target_i, target_j]

                coast_id[i, j] = id0

    return coast_id

##############################################################################
# Parameters #################################################################
##############################################################################

length_scale = 50.  # e-folding scale (lambda) for plastic (km)
efs          = 1/length_scale

cr           = 0.15 # proportion of mismanaged plastic waste generated in
                    # coastal regions that enters the ocean

##############################################################################
# File locations #############################################################
##############################################################################

# Directories
script_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
grid_dir   = script_dir + 'CMEMS/'
data_dir   = script_dir + 'PLASTIC_DATA/'

# Grid from CMEMS
grid_fh    = grid_dir + 'globmask.nc'

# Lebreton & Andrady plastic waste generation data
# https://www.nature.com/articles/s41599-018-0212-7
plastic_fh = data_dir + 'LebretonAndrady2019_MismanagedPlasticWaste.tif'

# Country ID grid(UN WPP-Adjusted Population Density v4.11)
# https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/sets/browse
id_fh      = data_dir + 'gpw_v4_national_identifier_grid_rev11_30_sec.tif'

# Meijer et al river plastic data
# https://advances.sciencemag.org/content/7/18/eaaz5803/tab-figures-data
river_fh   = data_dir + 'Meijer2021_midpoint_emissions.zip'

# Output netcdf fh
out_fh     = grid_dir + 'plastic_flux.nc'

##############################################################################
# Coastal plastics ###########################################################
##############################################################################

# Methodology
# 1. Upscale plastic/id data to 1/12 grid
# 2. Calculate nearest coastal cell on CMEMS grid
# 3. Calculate the distance of each cell to coastal cell with Haversine
# 4. Calculate the waste flux at each coastal cell (sum of weighted land cells)

# Load CMEMS grid
with Dataset(grid_fh, mode='r') as nc:
    # Cell boundaries
    lon_bnd = nc.variables['lon_rho'][:]
    lon_bnd = np.append(lon_bnd, 180)
    lat_bnd = nc.variables['lat_rho'][:]

    # Cell centres
    lon     = nc.variables['lon_psi'][:]
    lat     = nc.variables['lat_psi'][:]
    lon, lat = np.meshgrid(lon, lat)

    # Coast and lsm
    coast   = nc.variables['coast_psi'][:]
    lsm     = nc.variables['lsm_psi'][:]

# Load plastic data
plastic_data_obj = gdal.Open(plastic_fh)
plastic_data = plastic_data_obj.ReadAsArray()[:-1200, :]

# Load country codes
country_id_obj = gdal.Open(id_fh)
country_id = country_id_obj.ReadAsArray()[:-1200, :]

# 1. Upscale data to 1/12 grid
plastic_data = block_reduce(plastic_data, (10,10), np.sum)
plastic_data = np.float64(np.flipud(plastic_data))
country_id = block_reduce(country_id, (10,10), np.max)
country_id = np.flipud(country_id)

# 2. Calculate nearest coastal cells
cnearest = distance_transform_edt(1-coast,
                                  return_distances=False,
                                  return_indices=True)


# 3. Calculate the distance of each waste flux cell to the nearest coastal
#    cell
cdist = cdist_calc(cnearest, lon, lat, efs)

#    Modify cells by W_mod = W*exp(-efs*cdist)*cr
plastic_data *= cr*np.exp(-efs*cdist)

# 4. Calculate the waste flux at each coastal cell
cplastic = cplastic_calc(plastic_data, cnearest)

##############################################################################
# Riverine plastics ##########################################################
##############################################################################

river_file = gpd.read_file(river_fh)
river_data = np.array(river_file['dots_exten'])*1000.  # tons -> kg
river_lon  = np.array(river_file['geometry'].x)
river_lat  = np.array(river_file['geometry'].y)

# Bin onto the 1/12 degree grid
rplastic   = np.histogram2d(river_lon, river_lat,
                            bins=[lon_bnd, lat_bnd],
                            weights=river_data,
                            normed=False)[0].T

# Shift to coastal cells
rplastic   = cplastic_calc(rplastic, cnearest)


##############################################################################
# Label coastal cells ########################################################
##############################################################################

# Label coastal cells (coast) with their ISO country identifier
# Calculate the nearest country_id to each grid cell without a country_id
# A coastal cell with a country id takes that country id, otherwise it takes
# the nearest country id

country_id_lsm = np.copy(country_id)
country_id_lsm[country_id_lsm >= 0] = 0
country_id_lsm[country_id_lsm <  0] = 1

country_id_nearest = distance_transform_edt(country_id_lsm,
                                            return_distances=False,
                                            return_indices=True)

coast_id = id_coast_cells(coast, country_id, country_id_nearest)

##############################################################################
# Save to netcdf #############################################################
##############################################################################

total_coastal_plastic = '{:3.2f}'.format(np.sum(cplastic)/1e9) + ' Mt yr-1'
total_riverine_plastic = '{:3.2f}'.format(np.sum(rplastic)/1e9) + ' Mt yr-1'

with Dataset(out_fh, mode='w') as nc:
    # Create dimensions
    nc.createDimension('lon', lon.shape[1])
    nc.createDimension('lat', lat.shape[0])

    # Create variables
    nc.createVariable('lon', 'f4', ('lon'), zlib=True)
    nc.createVariable('lat', 'f4', ('lat'), zlib=True)
    nc.createVariable('coast_plastic', 'f8', ('lat', 'lon'), zlib=True)
    nc.createVariable('river_plastic', 'f8', ('lat', 'lon'), zlib=True)
    nc.createVariable('coast_id', 'i4', ('lat', 'lon'), zlib=True)
    nc.createVariable('coast', 'i4', ('lat', 'lon'), zlib=True)
    nc.createVariable('lsm', 'i4', ('lat', 'lon'), zlib=True)

    # Write variables
    nc.variables['lon'].long_name = 'longitude'
    nc.variables['lon'].units = 'degrees_east'
    nc.variables['lon'].standard_name = 'longitude'
    nc.variables['lon'][:] = lon[0, :]

    nc.variables['lat'].long_name = 'latitude'
    nc.variables['lat'].units = 'degrees_north'
    nc.variables['lat'].standard_name = 'latitude'
    nc.variables['lat'][:] = lat[:, 0]

    nc.variables['coast_plastic'].long_name = 'plastic_flux_at_coast_from_coastal_sources'
    nc.variables['coast_plastic'].units = 'kg yr-1'
    nc.variables['coast_plastic'].standard_name = 'coast_plastic'
    nc.variables['coast_plastic'].total_flux = total_coastal_plastic
    nc.variables['coast_plastic'][:] = cplastic

    nc.variables['river_plastic'].long_name = 'plastic_flux_at_coast_from_riverine_sources'
    nc.variables['river_plastic'].units = 'kg yr-1'
    nc.variables['river_plastic'].standard_name = 'river_plastic'
    nc.variables['river_plastic'].total_flux = total_riverine_plastic
    nc.variables['river_plastic'][:] = rplastic

    nc.variables['coast_id'].long_name = 'ISO_3166-1_numeric_code_of_coastal_cells'
    nc.variables['coast_id'].units = 'no units'
    nc.variables['coast_id'].standard_name = 'coast_id'
    nc.variables['coast_id'][:] = coast_id

    nc.variables['coast'].long_name = 'coast_mask'
    nc.variables['coast'].units = '1: coast, 0: not coast'
    nc.variables['coast'].standard_name = 'coast_mask'
    nc.variables['coast'][:] = coast

    nc.variables['lsm'].long_name = 'land_sea_mask'
    nc.variables['lsm'].units = 'no units'
    nc.variables['lsm'].standard_name = '1: land, 0: ocean'
    nc.variables['lsm'][:] = lsm

    # Global attributes
    date = datetime.now()
    date = date.strftime('%d/%m/%Y, %H:%M:%S')
    nc.date_created = date

    nc.country_id_source = 'https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/sets/browse'
    nc.coast_plastic_source = 'https://www.nature.com/articles/s41599-018-0212-7'
    nc.river_plastic_source = 'https://advances.sciencemag.org/content/7/18/eaaz5803/tab-figures-data'

