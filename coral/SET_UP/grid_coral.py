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
import cmasher as cmr
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from glob import glob
from osgeo import gdal, osr

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

    lsm_rho_w = nc.variables['mask_rho'][:]

with Dataset(fh['cmems'], mode='r') as nc:
    lon_psi_c = nc.variables['lon_psi'][:]
    lat_psi_c = nc.variables['lat_psi'][:]
    lon_rho_c = nc.variables['lon_rho'][:]
    lat_rho_c = nc.variables['lat_rho'][:]

    lsm_rho_c = 1-nc.variables['lsm_rho'][:]

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

    lsm_rho_c = lsm_rho_c[cmems_yi_range[0]:cmems_yi_range[1],
                          cmems_xi_range[0]:cmems_xi_range[1]]

coral_grid_w = np.zeros_like(lsm_rho_w, dtype=np.float64).T
coral_grid_w = coral_grid_w[1:-1, 1:-1]
coral_grid_c = np.zeros_like(lsm_rho_c, dtype=np.float64).T
coral_grid_c = coral_grid_c[1:-1, 1:-1]
###############################################################################
# Grid coral sites ############################################################
###############################################################################

# Loop through each file and bin all coral reef cells onto model grids

for i, reef_file in enumerate(fh['coral'][:]):
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

    img_lon, img_lat = np.meshgrid(lon_arr, lat_arr)

    # Now extract coordinates of reef cells
    reef_lon = np.ma.masked_array(img_lon, 1-reef_data).compressed()
    reef_lat = np.ma.masked_array(img_lat, 1-reef_data).compressed()

    # Now grid
    coral_grid_w += np.histogram2d(reef_lon, reef_lat, [lon_psi_w, lat_psi_w])[0]
    coral_grid_c += np.histogram2d(reef_lon, reef_lat, [lon_psi_c, lat_psi_c])[0]

    # reef_lon = np.linspace()
    print(i)

f, ax = plt.subplots(1, 1, figsize=(15, 10), constrained_layout=True)
ax.pcolormesh(lon_psi_w, lat_psi_w, coral_grid_w)




# # Load plastic data
# plastic_data_obj = gdal.Open(plastic_fh)
# plastic_data = plastic_data_obj.ReadAsArray()[:-1200, :]

# # Load country codes
# country_id_obj = gdal.Open(id_fh)
# country_id = country_id_obj.ReadAsArray()[:-1200, :]

# # 1. Upscale data to 1/12 grid
# plastic_data = block_reduce(plastic_data, (10,10), np.sum)
# plastic_data = np.float64(np.flipud(plastic_data))
# country_id = block_reduce(country_id, (10,10), np.max)
# country_id = np.flipud(country_id)
