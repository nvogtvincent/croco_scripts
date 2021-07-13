#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script grids plastic source fluxes onto the coastal grid
@Author: Noam Vogt-Vincent
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
import os.path
import numba
from numba.core import types
from numba.typed import Dict
from scipy.stats import mode

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
def plastic_conv(pop, pop_id, plastic_dict):
    n_y = np.shape(pop)[1]
    n_x = np.shape(pop)[2]
    n_t = np.shape(pop)[0]

    for i in range(n_y):
        for j in range(n_x):
            for k in range(n_t):
                iso_code = np.int16(pop_id[i, j])

                # Find corresponding plastic flux (kg per person per day)
                try:
                    flux = plastic_dict[iso_code]
                except:
                    flux = 0

                pop[k, i, j] *= flux

    return pop

@numba.jit(nopython=True)
def target_coords(nearest_cell, pop, pop_lon, pop_lat, lon, lat, efs):
    n_y = np.shape(nearest_cell)[1]
    n_x = np.shape(nearest_cell)[2]

    target_lon = np.zeros((n_y, n_x))
    target_lat = np.zeros_like(target_lon)

    coast_dist = np.zeros_like(target_lat)
    coast_plastic = np.zeros((np.shape(pop)[0],
                              np.shape(lon)[0],
                              np.shape(lon)[1]))

    for i in range(n_y):
        for j in range(n_x):
            # Only calculate if the population is nonzero
            if np.max(pop[:, i, j]) > 0:
                jidx = nearest_cell[1, i, j]
                iidx = nearest_cell[0, i, j]

                # Find coordinates of nearest coastal cell (CMEMS grid)
                target_lon = lon[iidx, jidx]
                target_lat = lat[iidx, jidx]

                # Find coordinates of current cell
                current_lon = pop_lon[i, j]
                current_lat = pop_lat[i, j]

                # Distance between cells
                dist = haversine_np(current_lon,
                                    current_lat,
                                    target_lon,
                                    target_lat)

                # Add the plastic flux to the target cell (on CMEMS grid)
                plastic_accum = pop[:, i, j]*np.exp(-efs*dist)
                if np.min(plastic_accum) < 0:
                    raise ValueError('Negative plastic encountered!')

                coast_plastic[:, iidx, jidx] += plastic_accum

    return coast_dist, coast_plastic

@numba.jit(nopython=True)
def id_coast_cells(coast, pop_id, nearest):
    coast_id = np.copy(coast)
    n_y = np.shape(coast_id)[0]
    n_x = np.shape(coast_id)[1]

    for i in range(n_y):
        for j in range(n_x):
            if coast[i, j] == 1:
                # Firstly check if the coast already has a country ID
                id0 = pop_id[i, j]

                if id0 == 32767:
                    iidx = pop_id_nearest[0, i, j]
                    jidx = pop_id_nearest[1, i, j]
                    id0 = pop_id[iidx, jidx]

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
script_dir = os.path.dirname(os.path.realpath(__file__))
grid_dir   = script_dir + '/CMEMS/'
data_dir   = script_dir + '/PLASTIC_DATA/'

# Grid from CMEMS
grid_fh    = grid_dir + 'globmask.nc'

# Jambeck plastic data
jambeck_fh = data_dir + 'jambeck_2015.csv'

# Population grids (UN WPP-Adjusted Population Density v4.11)
pop_raw_fh   = data_dir + 'gpw_v4_population_count_adjusted_rev11_2pt5_min.nc'
pop_fh       = data_dir + 'popdata.nc'
pop_var      = 'UN WPP-Adjusted Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes'
nt           = 5   # Number of time slices in pop data
ci           = 10  # Index of country labels


##############################################################################
# Coastal plastics ###########################################################
##############################################################################

# Methodology
# 1. Convert population to waste flux using figures from Jambeck 2015
# 2. Calculate nearest coastal cell on CMEMS grid and then downscale to 1/24 deg
# 3. Calculate the distance of each waste flux cell to that nearest coastal cell
# 4. Calculate the waste flux at each coastal cell (sum of weighted land cells)

# Load CMEMS grid
with Dataset(grid_fh, mode='r') as nc:
    # Cell boundaries
    lon_bnd = nc.variables['lon_rho'][:]
    lat_bnd = nc.variables['lat_rho'][:]

    # Cell centres
    lon     = nc.variables['lon_psi'][:]
    lat     = nc.variables['lat_psi'][:]

    ny = len(lat)
    nx = len(lon)

    # Coast and lsm
    coast   = nc.variables['coast_psi'][:]
    lsm     = nc.variables['lsm_psi'][:]

# Load population data
with Dataset(pop_raw_fh, mode='r') as nc:
    # Remove missing latitudes (CMEMS is for -80 to + 90)
    # i.e. remove first 20*24 entries
    pop_lon  = nc.variables['longitude'][:]
    pop_lat  = np.flip(nc.variables['latitude'][:-10*24])
    pop      = np.flip(nc.variables[pop_var][:nt, :-10*24, :], axis=1)
    pop_id   = np.flip(nc.variables[pop_var][ci, :-10*24, :], axis=0)

# 1. Calculate plastic fluxes
pdata = np.genfromtxt(jambeck_fh, delimiter=',', skip_header=True,
                      usecols=3)
np.append(pdata, 0) # Add 0 flux for ocean
pdata = np.float32(pdata) # kg/person/day

# ISO codes for countries
iso   = np.genfromtxt(jambeck_fh, delimiter=',', skip_header=True,
                      usecols=2)
np.append(pdata, 32767) # Add 0 flux for ocean
iso   = np.int16(iso)

# Turn into typed numba dict
pflux = Dict.empty(key_type=types.int16,
                    value_type=types.float32)

for i in range(len(pdata)):
    pflux[iso[i]] = pdata[i]

# Now convert population to plastic flux
pop = plastic_conv(pop.filled(0), pop_id, pflux)

# 2. Calculate nearest coastal cells
cell_dist, nearest_cell = distance_transform_edt(1-coast,
                                                   return_indices=True)

# And expand to 1/24deg
lon, lat = np.meshgrid(lon, lat)
pop_lon, pop_lat = np.meshgrid(pop_lon, pop_lat)
target_lon = np.zeros_like(pop_lon)    # Seperate arrays for coordinates of
                                       # target coastal cell
target_lat = np.zeros_like(pop_lat)

cell_dist = np.repeat(cell_dist, 2, axis=0)
cell_dist = np.repeat(cell_dist, 2, axis=1)
nearest_cell = np.repeat(nearest_cell, 2, axis=1)
nearest_cell = np.repeat(nearest_cell, 2, axis=2)
target_lon = np.repeat(target_lon, 2, axis=0)
target_lon = np.repeat(target_lon, 2, axis=1)
target_lat = np.repeat(target_lat, 2, axis=0)
target_lat = np.repeat(target_lat, 2, axis=1)

# 3. Calculate the distance of each waste flux cell to the nearest coastal
#    cell
# 4. Calculate the waste flux at each coastal cell (sum of weighted land cells)

coast_dist, coast_plastic = target_coords(nearest_cell, pop,
                                          pop_lon, pop_lat,
                                          lon, lat,
                                          efs)
# Include conversion rate and convert to metric tons
coast_plastic *= cr/1000.

##############################################################################
# Label coastal cells ########################################################
##############################################################################

# Label coastal cells (coast) with their ISO country identifier
# To do this, we we shrink the pop_id grid to the size of the CMEMS grid and
# then calculate the nearest country_id to each grid cell without a country_id
# A coastal cell with a country id takes that country id, otherwise it takes
# the nearest country id

pop_id_red = block_reduce(pop_id, (2,2), np.min) # min to avoid ocean attr.

pop_id_lsm = np.copy(pop_id_red)
pop_id_lsm[pop_id_lsm < 32767] = 0
pop_id_lsm[pop_id_lsm > 0]     = 1

pop_id_nearest = distance_transform_edt(pop_id_lsm,
                                        return_distances=False,
                                        return_indices=True)

coast_id = id_coast_cells(coast, pop_id_red, pop_id_nearest)


# f, a0 = plt.subplots(1, 1, figsize=(20, 10))
# a0.imshow(np.flipud(coast_plastic[0, :, :]), vmax=1e3)
# plt.savefig('test1.png', dpi=300)

f, a0 = plt.subplots(1, 1, figsize=(20, 10))
a0.imshow(np.flipud(coast_id))
plt.savefig('test2.png', dpi=300)

# f, a0 = plt.subplots(1, 1, figsize=(20, 10))
# lon = lon.flatten()
# lat = lat.flatten()
# coast_plastic = coast_plastic[0, :, :].flatten()

# lon = np.delete(lon, (coast_plastic == 0))
# lat = np.delete(lat, (coast_plastic == 0))
# coast_plastic = np.delete(coast_plastic, (coast_plastic == 0))


# a0.scatter(lon, lat, s=coast_plastic/2e3, vmin=0, vmax=1)

# plt.savefig('test3.png', dpi=300)

# #cell_dist, nearest_cell = distance_transform_edt(pop_lsm, return_indices=True)
# print('test)')

