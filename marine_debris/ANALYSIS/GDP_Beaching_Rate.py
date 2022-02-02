#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script extracts a beaching rate from GDP data
@author: Noam Vogt-Vincent

@drifter_source: https://www.aoml.noaa.gov/phod/gdp/interpolated/data/all.php
@coastline_source: https://www.soest.hawaii.edu/pwessel/gshhg/
@bathymetry_source: https://www.gebco.net/data_and_products/gridded_bathymetry_data/

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
import xarray as xr
from osgeo import gdal, osr
from glob import glob
from osgeo import gdal
from tqdm import tqdm
from netCDF4 import Dataset, num2date
from shapely.geometry import Point


# Methodology:
# 1. Determine whether a trajectory ever approaches within 1/12 of the coast
# 2. Calculate the cumulative time the drifter spends within 1/12 of the coast
# 3. Determine whether drifter has beached, using the following two criteria:
#    a. Last drifter location is within 500m of the GSHHG coast
#    b. Last drifter location is in <30m water depth (GEBCO2021)
# 4. Calculate F(beach) as a function of cumulative time within 1/12 of the coast

# PARAMETERS
param = {'out_name': 'fisheries_waste_flux_'}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/',
        'gdp': os.path.dirname(os.path.realpath(__file__)) + '/../GDP_DATA/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'cmems_grid': dirs['grid'] + 'griddata.nc',
      'gdp': sorted(glob(dirs['gdp'] + 'bd*.dat')),
      'gebco': dirs['grid'] + 'LOC/gebco_2021/GEBCO_2021.nc',
      'coast_005deg': dirs['grid'] + 'LOC/coastal_mask/coastal_mask_005.gpkg',
      'coast_083deg': dirs['grid'] + 'LOC/coastal_mask/GSHHS_h_L1_buffer083_res01.tif',
      'fig':   dirs['fig'] + 'GDP_beaching_rate.png',}

###############################################################################
# LOAD DATA ###################################################################
###############################################################################

# Load LSM + coastal cells ####################################################
with Dataset(fh['cmems_grid'], mode='r') as nc:
    lon_bnd = nc.variables['lon_rho'][:]
    lat_bnd = nc.variables['lat_rho'][:]
    lon_bnd = np.concatenate((lon_bnd, [180]))

    lsm = nc.variables['iso_psi_all'][:]
    lsm_cst = nc.variables['iso_psi'][:]
    lsm[lsm > 0] = 1
    lsm_cst[lsm_cst > 0] = 1

# Load coarse + rasterised 0.083deg coastal mask ##############################
coast83_obj = gdal.Open(fh['coast_083deg'])
coast83 = np.flipud(coast83_obj.ReadAsArray()[:])

# Coordinate system
xdim = coast83_obj.RasterXSize
ydim = coast83_obj.RasterYSize
gt = coast83_obj.GetGeoTransform()
x_lim = [gt[0], gt[0] + xdim*gt[1] + ydim*gt[2]]
y_lim = [gt[3], gt[3] + xdim*gt[4] + ydim*gt[5]]
gt_res = gt[1]

lon_c83 = np.linspace(x_lim[0], x_lim[1], num=xdim)
lat_c83 = np.linspace(y_lim[0], y_lim[1], num=ydim)[::-1]

lon_c83_bnd = np.concatenate((lon_c83, [lon_c83[-1]+gt[1]]))-gt[1]/2
lat_c83_bnd = np.concatenate((lat_c83, [lat_c83[-1]+gt[1]]))-gt[1]/2

coast83 = xr.open_rasterio(fh['coast_083deg'])

# Load 0.005deg (500m) coastal mask ###########################################
coast5_obj = gpd.read_file(fh['coast_005deg'])

# Load GEBCO bathymetry #######################################################
bath_data = xr.open_dataset(fh['gebco'])

###############################################################################
# ANALYSE GDP TRAJECTORIES ####################################################
###############################################################################
for gdp_fh in fh['gdp']:
    # Read file
    df = pd.read_csv(gdp_fh, sep='\s+', header=None,
                     usecols=[0, 4, 5, 9],
                     names=['ID', 'lat', 'lon', 'vel'],)

    # Change longitude from 0-360 -> -180-180
    lon_ = df['lon'].values
    lon_[lon_ > 180] = lon_[lon_ > 180]-360
    df['lon'] = lon_

    # Extract a list of drifter IDs
    drifter_id_list = np.unique(df['ID'])

    # Loop through drifters
    for drifter_id in drifter_id_list:
        df_drifter = df.loc[df['ID'] == drifter_id].copy()

        # Carry out an initial sweep to identify if there is any overlap with
        # coastal cells
        gridded_drifter_loc = np.histogram2d(df_drifter['lon'].values,
                                             df_drifter['lat'].values,
                                             bins=[lon_c83_bnd, lat_c83_bnd])[0].T

        if np.sum(gridded_drifter_loc*coast83):
            # Assess whether points are within 1/12 degree of coast
            coast_status = np.zeros((len(df_drifter)))

            for i in tqdm(range(len(df_drifter)), total=len(df_drifter)):
                lon_idx = np.searchsorted(lon_c83_bnd, df_drifter.iloc[i]['lon'])-1
                lat_idx = np.searchsorted(lat_c83_bnd, df_drifter.iloc[i]['lat'])-1
                coast_status[i] = 1 if coast83[lat_idx, lon_idx] == 1 else 0

                # pos = Point(df_drifter.iloc[i]['lon'], df_drifter.iloc[i]['lat'])
                # within_coast = coast.geometry.intersects(pos)[0]
                # coast_status = 1 if within_coast else 0

            df_drifter['coast_status'] = coast_status

            # if np.sum(coast_status):
            #     print()

            f, a0 = plt.subplots(1, 1, figsize=(10, 10),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

            # a0.pcolormesh(lon_cm_bnd, lat_cm_bnd, coast_cm, cmap=cmr.neutral)
            a0.scatter(df_drifter['lon'], df_drifter['lat'], s=1)
            a0.scatter(df_drifter['lon'].loc[df_drifter['coast_status'] == 1],
                       df_drifter['lat'].loc[df_drifter['coast_status'] == 1],
                       c=df_drifter['coast_status'].loc[df_drifter['coast_status'] == 1],
                       vmin=0, vmax=1)
            a0.coastlines()
            a0.set_xlim([np.min(df_drifter['lon']-1), np.max(df_drifter['lon'])+1])
            a0.set_ylim([np.min(df_drifter['lat']-1), np.max(df_drifter['lat']+1)])
            plt.show()
            print()
        else:
            print('Rejected!')

        print()











