#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script regrids reef probability data to the CROCO and CMEMS grids
@author: Noam Vogt-Vincent

@reef_probability_source: https://doi.org/10.1007/s00338-020-02005-6
@eez_source: https://www.marineregions.org/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr
from tqdm import tqdm
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from shapely.geometry import Point
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import geopandas as gpd
from glob import glob
from osgeo import gdal, osr
from geographiclib.geodesic import Geodesic
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

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
        'loc': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/LOC/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'cmems': dirs['grid'] + 'griddata_cmems.nc',
      'winds': dirs['grid'] + 'griddata_winds.nc',
      'coral': sorted(glob(dirs['coral_data'] + '**/*.tif', recursive=True)),
      'eez': dirs['loc'] + 'EEZ_Land_v3_202030.shp',
      'coral_id': dirs['loc'] + 'test2.gpkg',
      'out':   dirs['grid'] + 'coral_grid.nc',
      'fig':   dirs['fig'] + 'reef_map',}

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
print('Gridding coral sites...')
for i, reef_file in tqdm(enumerate(fh['coral'][:]), total=len(fh['coral'])):

    # if i > 100: # TEMPORARY TIME SAVING HACK, REMOVE!
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

print('Shifting coral cells (part 1)...')
for yidx, xidx in tqdm(zip(land_idx_w[0], land_idx_w[1]), total=len(land_idx_w[0])):
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

print('Shifting coral cells (part 2)...')
for yidx, xidx in tqdm(zip(land_idx_c[0], land_idx_c[1]), total=len(land_idx_c[0])):
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
# Apply EEZ mask ##############################################################
###############################################################################

eez_file = gpd.read_file(fh['eez'])

# Create new grids
eez_grid_w = np.zeros_like(coral_grid_w, dtype=np.int16)
eez_grid_c = np.zeros_like(coral_grid_c, dtype=np.int16)

# Loop through all reef locations
coral_idx_w = np.where(coral_grid_w > 0)
coral_idx_c = np.where(coral_grid_c > 0)

print('Finding EEZ at points (part 1)...')
for yidx, xidx in tqdm(zip(coral_idx_w[0], coral_idx_w[1]), total=len(coral_idx_w[0])):
    pos = Point(lon_rho_w[xidx], lat_rho_w[yidx])

    # Check which EEZ the point intersects
    eez_intersection = eez_file.geometry.intersects(pos)
    eez_intersection = eez_intersection[eez_intersection == True]

    if len(eez_intersection) != 1:
        raise NotImplementedError('EEZ not found at location ' + lon_rho_w[xidx] + 'E, ' + lat_rho_w[yidx] + 'N!')

    # Find the ISO code for that EEZ
    iso_code = eez_file['UN_SOV1'][eez_intersection.index[0]]

    # Uniquely identify Chagos (not making any suggestion about sovereignty, just to distinguish)
    if iso_code == 480:
        if lon_rho_w[xidx] > 65:
            iso_code = 86

    # Add to grid
    eez_grid_w[yidx, xidx] = iso_code

print('Finding EEZ at points (part 2)...')
for yidx, xidx in tqdm(zip(coral_idx_c[0], coral_idx_c[1]), total=len(coral_idx_c[0])):
    pos = Point(lon_rho_c[xidx], lat_rho_c[yidx])

    # Check which EEZ the point intersects
    eez_intersection = eez_file.geometry.intersects(pos)
    eez_intersection = eez_intersection[eez_intersection == True]

    if len(eez_intersection) != 1:
        raise NotImplementedError('EEZ not found at location ' + lon_rho_c[xidx] + 'E, ' + lat_rho_c[yidx] + 'N!')

    # Find the ISO code for that EEZ
    iso_code = eez_file['UN_SOV1'][eez_intersection.index[0]]

    # Uniquely identify Chagos (not making any suggestion about sovereignty, just to distinguish)
    if iso_code == 480:
        if lon_rho_w[xidx] > 65:
            iso_code = 86

    # Add to grid
    eez_grid_c[yidx, xidx] = iso_code

###############################################################################
# Clustering ##################################################################
###############################################################################

# See https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html
# or https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
# WIP!!
n_clusters=30
lon_w, lat_w = np.meshgrid(lon_rho_w, lat_rho_w)
lon_w = lon_w[eez_grid_w == 690]
lat_w = lat_w[eez_grid_w == 690]
X = np.concatenate((lon_w, lat_w)).reshape((2,-1)).T
knn_graph = kneighbors_graph(X, n_clusters, include_self=False)
linkage='ward'
model = AgglomerativeClustering(linkage=linkage, connectivity=knn_graph, n_clusters=n_clusters)
model.fit(X)
data_crs = ccrs.PlateCarree()

f, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True,
                     subplot_kw={'projection': ccrs.PlateCarree()})
ax.scatter(X[:,0], X[:,1], c=model.labels_, s=1, transform=data_crs)
# ax.set_ylim([-17, -9])


###############################################################################
# Plot reef sites #############################################################
###############################################################################

# Plot 'before'
f, ax = plt.subplots(1, 1, figsize=(24, 10), constrained_layout=True,
                     subplot_kw={'projection': ccrs.PlateCarree()})

data_crs = ccrs.PlateCarree()
oceanc = ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(lsm_rho_w == 1, coral_grid_w_preshift)[1:-1, 1:-1],
                       norm=colors.LogNorm(vmin=1e2, vmax=1e8), cmap=cmr.flamingo_r, transform=data_crs)
landc = ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(lsm_rho_w == 0, coral_grid_w_preshift)[1:-1, 1:-1],
                      norm=colors.LogNorm(vmin=1e2, vmax=1e8), cmap=cmr.freeze_r, transform=data_crs)
ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(coral_grid_w_preshift > 0, 1-lsm_rho_w)[1:-1, 1:-1],
              vmin=-2, vmax=1, cmap=cmr.neutral, transform=data_crs)

gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', linestyle='-')
gl.xlocator = mticker.FixedLocator(np.arange(35, 95, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-25, 5, 5))
gl.ylabels_right = False
gl.xlabels_top = False

ax.set_xlim([34.62, 77.5])
ax.set_ylim([-23.5, 0])
ax.spines['geo'].set_linewidth(1)
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_title('Coral cells on WINDS grid (preproc)')

cax1 = f.add_axes([ax.get_position().x1+0.07,ax.get_position().y0-0.10,0.015,ax.get_position().height+0.196])
cax2 = f.add_axes([ax.get_position().x1+0.12,ax.get_position().y0-0.10,0.015,ax.get_position().height+0.196])

cb1 = plt.colorbar(oceanc, cax=cax1, pad=0.1)
cb1.set_label('Coral surface area in ocean cell (m2)', size=12)
cb2 = plt.colorbar(landc, cax=cax2, pad=0.1)
cb2.set_label('Coral surface area in land cell (m2)', size=12)

ax.set_aspect('equal', adjustable=None)
ax.margins(x=-0.01, y=-0.01)

plt.savefig(fh['fig'] + '_WINDS_preproc.png', dpi=300)

f, ax = plt.subplots(1, 1, figsize=(24, 10), constrained_layout=True,
                     subplot_kw={'projection': ccrs.PlateCarree()})

data_crs = ccrs.PlateCarree()
oceanc = ax.pcolormesh(lon_rho_c, lat_rho_c, np.ma.masked_where(lsm_psi_c == 1, coral_grid_c_preshift),
                       norm=colors.LogNorm(vmin=1e2, vmax=1e9), cmap=cmr.flamingo_r, transform=data_crs)
landc = ax.pcolormesh(lon_rho_c, lat_rho_c, np.ma.masked_where(lsm_psi_c == 0, coral_grid_c_preshift),
                      norm=colors.LogNorm(vmin=1e2, vmax=1e9), cmap=cmr.freeze_r, transform=data_crs)
ax.pcolormesh(lon_rho_c, lat_rho_c, np.ma.masked_where(coral_grid_c_preshift > 0, 1-lsm_psi_c),
              vmin=-2, vmax=1, cmap=cmr.neutral, transform=data_crs)

gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', linestyle='-')
gl.xlocator = mticker.FixedLocator(np.arange(35, 95, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-25, 5, 5))
gl.ylabels_right = False
gl.xlabels_top = False

ax.set_xlim([34.62, 77.5])
ax.set_ylim([-23.5, 0])
ax.spines['geo'].set_linewidth(1)
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_title('Coral cells on CMEMS grid (preproc)')

cax1 = f.add_axes([ax.get_position().x1+0.07,ax.get_position().y0-0.10,0.015,ax.get_position().height+0.196])
cax2 = f.add_axes([ax.get_position().x1+0.12,ax.get_position().y0-0.10,0.015,ax.get_position().height+0.196])

cb1 = plt.colorbar(oceanc, cax=cax1, pad=0.1)
cb1.set_label('Coral surface area in ocean cell (m2)', size=12)
cb2 = plt.colorbar(landc, cax=cax2, pad=0.1)
cb2.set_label('Coral surface area in land cell (m2)', size=12)

ax.set_aspect('equal', adjustable=None)
ax.margins(x=-0.01, y=-0.01)

plt.savefig(fh['fig'] + '_CMEMS_preproc.png', dpi=300)

# Plot 'after'
f, ax = plt.subplots(1, 1, figsize=(24, 10), constrained_layout=True,
                     subplot_kw={'projection': ccrs.PlateCarree()})

data_crs = ccrs.PlateCarree()
coral = ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(coral_grid_w == 0, coral_grid_w)[1:-1, 1:-1],
                       norm=colors.LogNorm(vmin=1e2, vmax=1e8), cmap=cmr.flamingo_r, transform=data_crs)
ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(lsm_rho_w == 0, 1-lsm_rho_w)[1:-1, 1:-1],
              vmin=-2, vmax=1, cmap=cmr.neutral, transform=data_crs)

gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', linestyle='-')
gl.xlocator = mticker.FixedLocator(np.arange(35, 95, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-25, 5, 5))
gl.ylabels_right = False
gl.xlabels_top = False

ax.set_xlim([34.62, 77.5])
ax.set_ylim([-23.5, 0])
ax.spines['geo'].set_linewidth(1)
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_title('Coral cells on WINDS grid (postproc)')

cax1 = f.add_axes([ax.get_position().x1+0.07,ax.get_position().y0-0.10,0.015,ax.get_position().height+0.196])

cb1 = plt.colorbar(oceanc, cax=cax1, pad=0.1)
cb1.set_label('Coral surface area in cell (m2)', size=12)

ax.set_aspect('equal', adjustable=None)
ax.margins(x=-0.01, y=-0.01)

plt.savefig(fh['fig'] + '_WINDS_postproc.png', dpi=300)

f, ax = plt.subplots(1, 1, figsize=(24, 10), constrained_layout=True,
                     subplot_kw={'projection': ccrs.PlateCarree()})

data_crs = ccrs.PlateCarree()
coral = ax.pcolormesh(lon_rho_c, lat_rho_c, np.ma.masked_where(lsm_psi_c == 1, coral_grid_c),
                       norm=colors.LogNorm(vmin=1e2, vmax=1e9), cmap=cmr.flamingo_r, transform=data_crs)
ax.pcolormesh(lon_rho_c, lat_rho_c, np.ma.masked_where(lsm_psi_c == 0, 1-lsm_psi_c),
              vmin=-2, vmax=1, cmap=cmr.neutral, transform=data_crs)

gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', linestyle='-')
gl.xlocator = mticker.FixedLocator(np.arange(35, 95, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-25, 5, 5))
gl.ylabels_right = False
gl.xlabels_top = False

ax.set_xlim([34.62, 77.5])
ax.set_ylim([-23.5, 0])
ax.spines['geo'].set_linewidth(1)
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_title('Coral cells on CMEMS grid (postproc)')

cax1 = f.add_axes([ax.get_position().x1+0.07,ax.get_position().y0-0.10,0.015,ax.get_position().height+0.196])

cb1 = plt.colorbar(oceanc, cax=cax1, pad=0.1)
cb1.set_label('Coral surface area in cell (m2)', size=12)

ax.set_aspect('equal', adjustable=None)
ax.margins(x=-0.01, y=-0.01)

plt.savefig(fh['fig'] + '_CMEMS_postproc.png', dpi=300)


