#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script regrids reef probability data to the CROCO and CMEMS grids
@author: Noam Vogt-Vincent
@reef_probability_source: https://doi.org/10.1007/s00338-020-02005-6
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from glob import glob
from osgeo import gdal, osr
from geographiclib.geodesic import Geodesic

###############################################################################
# Parameters ##################################################################
###############################################################################

# PARAMETERS
param = {'grid_res': 1.0,                          # Grid resolution in degrees
         'lon_range': [20, 100],                   # Longitude range for output
         'lat_range': [-40, 30],                   # Latitude range for output

         'dict_name': 'mmsi_dict.pkl',
         'out_name': 'fisheries_waste_flux_'}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/',
        'coral_data': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/CORAL/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'cmems': dirs['grid'] + 'griddata_cmems.nc',
      'winds': dirs['grid'] + 'griddata_winds.nc',
      'coral': sorted(glob(dirs['coral_data'] + '**/*.tif', recursive=True)),
      'out':   dirs['grid'] + 'coral_grid.nc',
      'fig':   dirs['fig'] + 'reef_map_',}

###############################################################################
# Open grids ##################################################################
###############################################################################

with Dataset(fh['winds'], mode='r') as nc:
    lon_psi_w = nc.variables['lon_psi'][0, :]
    lat_psi_w = nc.variables['lat_psi'][:, 0]
    lon_rho_w = nc.variables['lon_rho'][0, :]
    lat_rho_w = nc.variables['lat_rho'][:, 0]

    lsm_rho_w = 1-nc.variables['mask_rho'][:].astype(np.int64)

with Dataset(fh['cmems'], mode='r') as nc:
    lon_psi_c = nc.variables['lon_psi'][:]
    lat_psi_c = nc.variables['lat_psi'][:]
    lon_rho_c = nc.variables['lon_rho'][:]
    lat_rho_c = nc.variables['lat_rho'][:]

    lsm_psi_c = nc.variables['lsm_rho'][:].astype(np.int64)

    # Constrain to same domain as WINDS
    lon_range = [lon_rho_w[0], lon_rho_w[-1]]
    lat_range = [lat_rho_w[0], lat_rho_w[-1]]

    cmems_xi_range = [np.where(lon_rho_c < lon_range[0])[0][-1],
                          np.where(lon_rho_c > lon_range[1])[0][0]]
    cmems_yi_range = [np.where(lat_rho_c < lat_range[0])[0][-1],
                          np.where(lat_rho_c > lat_range[1])[0][0]]

    lon_psi_c = lon_psi_c[cmems_xi_range[0]:cmems_xi_range[1]-1]
    lat_psi_c = lat_psi_c[cmems_yi_range[0]:cmems_yi_range[1]-1]

    lon_rho_c = lon_rho_c[cmems_xi_range[0]:cmems_xi_range[1]]
    lat_rho_c = lat_rho_c[cmems_yi_range[0]:cmems_yi_range[1]]

    lsm_psi_c = lsm_psi_c[cmems_yi_range[0]:cmems_yi_range[1]-1,
                          cmems_xi_range[0]:cmems_xi_range[1]-1]

# Note: CMEMS is given on a A-grid (i.e. rho points) whereas WINDS is given on
# a C-grid (i.e. u/v points ~ psi), so define coral points on psi cells for
# CMEMS and rho cells for WINDS. We give the WINDS grid a border of 1 so we
# exclude reef cells outside of the active domain.
coral_grid_w = np.zeros_like(lsm_rho_w, dtype=np.float64).T
coral_grid_w = coral_grid_w[1:-1, 1:-1]
coral_grid_c = np.zeros_like(lsm_psi_c, dtype=np.float64)
coral_grid_c = np.zeros_like(lsm_psi_c, dtype=np.float64).T

###############################################################################
# Grid coral sites ############################################################
###############################################################################

# Loop through each file and bin all coral reef cells onto model grids
for i, reef_file in enumerate(fh['coral'][:]):

    # if i > 200: # TEMPORARY TIME SAVING HACK, REMOVE!
    #     break


    reef_data_obj = gdal.Open(reef_file)
    reef_data = reef_data_obj.ReadAsArray()[:]

    # Coordinate system
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(reef_data_obj.GetProjectionRef())
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    transform = osr.CoordinateTransformation(old_cs, new_cs)

    xdim = reef_data_obj.RasterXSize
    ydim = reef_data_obj.RasterYSize
    gt = reef_data_obj.GetGeoTransform()
    x_lim = [gt[0], gt[0] + xdim*gt[1] + ydim*gt[2]]
    y_lim = [gt[3], gt[3] + xdim*gt[4] + ydim*gt[5]]

    img_lon_lim = [transform.TransformPoint(x_lim[0], y_lim[0])[1],
                   transform.TransformPoint(x_lim[1], y_lim[0])[1]]
    img_lat_lim = [transform.TransformPoint(x_lim[0], y_lim[0])[0],
                   transform.TransformPoint(x_lim[0], y_lim[1])[0]]

    lon_arr = np.linspace(img_lon_lim[0], img_lon_lim[1], num=xdim)
    lat_arr = np.linspace(img_lat_lim[0], img_lat_lim[1], num=ydim)

    # Now calculate the (approximate) surface area per cell
    dist_x = Geodesic.WGS84.Inverse(lat_arr[0], lon_arr[0], lat_arr[0], lon_arr[1])['s12']
    dist_y = Geodesic.WGS84.Inverse(lat_arr[0], lon_arr[0], lat_arr[1], lon_arr[0])['s12']
    SA = dist_x*dist_y

    img_lon, img_lat = np.meshgrid(lon_arr, lat_arr)

    # Now extract coordinates of reef cells
    reef_lon = np.ma.masked_array(img_lon, 1-reef_data).compressed()
    reef_lat = np.ma.masked_array(img_lat, 1-reef_data).compressed()

    # Now grid
    coral_grid_w += np.histogram2d(reef_lon, reef_lat, [lon_psi_w, lat_psi_w])[0]*SA
    coral_grid_c += np.histogram2d(reef_lon, reef_lat, [lon_rho_c, lat_rho_c])[0]*SA

###############################################################################
# Shift reefs on land #########################################################
###############################################################################

# Re-insert the WINDS grid into the full grid
coral_grid_w = np.pad(coral_grid_w, 1, mode='constant', constant_values=0)
coral_grid_w = coral_grid_w.astype(np.int32).T
coral_grid_c = coral_grid_c.astype(np.int32).T

# Now assess overlap with the land mask and shift reefs to adjacent cells
# if necessary
coral_grid_w_overlap = coral_grid_w*lsm_rho_w
coral_grid_c_overlap = coral_grid_c*lsm_psi_c

land_idx_w = np.where(coral_grid_w_overlap > 0)
land_idx_c = np.where(coral_grid_c_overlap > 0)

coral_area_before_w = np.sum(coral_grid_w)
coral_area_before_c = np.sum(coral_grid_c)

coral_grid_w_preshift = np.copy(coral_grid_w)
coral_grid_c_preshift = np.copy(coral_grid_c)

for yidx, xidx in zip(land_idx_w[0], land_idx_w[1]):
    ocean_found = False
    i = 0
    coral_area_in_land_cell = coral_grid_w[yidx, xidx]

    # RESTRICT TO i <= 2??
    while ocean_found == False:
        i += 1 # Expand search radius
        local_zone = lsm_rho_w[yidx-i:yidx+i+1, xidx-i:xidx+i+1]
        n_ocean_cells = np.sum(1-local_zone)

        if n_ocean_cells:
            ocean_found = True

    coral_grid_w[yidx-i:yidx+i+1, xidx-i:xidx+i+1] += (1-local_zone)*int(coral_area_in_land_cell/n_ocean_cells)
    coral_grid_w[yidx, xidx] = 0

for yidx, xidx in zip(land_idx_c[0], land_idx_c[1]):
    ocean_found = False
    i = 0
    coral_area_in_land_cell = coral_grid_c[yidx, xidx]

    while ocean_found == False:
        i += 1 # Expand search radius
        local_zone = lsm_psi_c[yidx-i:yidx+i+1, xidx-i:xidx+i+1]
        n_ocean_cells = np.sum(1-local_zone)

        if n_ocean_cells:
            ocean_found = True

    coral_grid_c[yidx-i:yidx+i+1, xidx-i:xidx+i+1] += (1-local_zone)*int(coral_area_in_land_cell/n_ocean_cells)
    coral_grid_c[yidx, xidx] = 0

coral_area_after_w = np.sum(coral_grid_w)
coral_area_after_c = np.sum(coral_grid_c)

###############################################################################
# Plot reef sites #############################################################
###############################################################################

# Plot 'before'
f, ax = plt.subplots(1, 1, figsize=(15, 10), constrained_layout=True)
ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(lsm_rho_w == 1, coral_grid_w_preshift)[1:-1, 1:-1],
              norm=colors.LogNorm(vmin=1e3, vmax=1e7), cmap=cmr.flamingo_r)
ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(lsm_rho_w == 0, coral_grid_w_preshift)[1:-1, 1:-1],
              norm=colors.LogNorm(vmin=1e3, vmax=1e7), cmap=cmr.ocean_r)
ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(coral_grid_w_preshift > 0, 1-lsm_rho_w)[1:-1, 1:-1],
              vmin=-2, vmax=1, cmap=cmr.neutral)
# f, ax = plt.subplots(1, 1, figsize=(15, 10), constrained_layout=True)
# ax.pcolormesh(lon_psi_w, lat_psi_w, coral_grid_w[1:-1, 1:-1])






# f, ax = plt.subplots(1, 1, figsize=(15, 10), constrained_layout=True)
# ax.pcolormesh(lon_rho_c, lat_rho_c, coral_grid_c)

