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
from netCDF4 import Dataset
from shapely.geometry import Point, Polygon
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import geopandas as gpd
from glob import glob
from osgeo import gdal, osr
from shapely.geometry.polygon import orient
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import KDTree

###############################################################################
# Parameters ##################################################################
###############################################################################

# PARAMETERS
param = {'av_cells_per_cluster': 40}

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

    SA = 9.55**2 # (see https://allencoralatlas.org/methods/)

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

# Calculate surface area of each grid cell
coral_idx_w = np.where(coral_grid_w > 0)
coral_idx_c = np.where(coral_grid_c > 0)

cell_SA_w = np.zeros_like(coral_grid_w)
cell_SA_c = np.zeros_like(coral_grid_c)
coral_frac_w = np.zeros_like(coral_grid_w)
coral_frac_c = np.zeros_like(coral_grid_c)

print('Calculating grid surface area (part 1)...')
# See https://gis.stackexchange.com/questions/413349/calculating-area-of-lat-lon-polygons-without-transformation-using-geopandas
# for approach

for yidx, xidx in tqdm(zip(coral_idx_w[0], coral_idx_w[1]), total=len(coral_idx_w[0])):
    x0 = lon_psi_w[xidx-1]
    x1 = lon_psi_w[xidx]
    y0 = lat_psi_w[yidx-1]
    y1 = lat_psi_w[yidx]

    cell_geom = gpd.GeoSeries(Polygon([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])).set_crs('epsg:4326')
    cell_geod = cell_geom.crs.get_geod()
    cell_area = int(cell_geod.geometry_area_perimeter(orient(cell_geom[0], 1))[0])

    cell_SA_w[yidx, xidx] = cell_area

print('Calculating grid surface area (part 2)...')
for yidx, xidx in tqdm(zip(coral_idx_c[0], coral_idx_c[1]), total=len(coral_idx_c[0])):
    x0 = lon_rho_c[xidx]
    x1 = lon_rho_c[xidx+1]
    y0 = lat_rho_c[yidx]
    y1 = lat_rho_c[yidx+1]

    cell_geom = gpd.GeoSeries(Polygon([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])).set_crs('epsg:4326')
    cell_geod = cell_geom.crs.get_geod()
    cell_area = int(cell_geod.geometry_area_perimeter(orient(cell_geom[0], 1))[0])

    cell_SA_c[yidx, xidx] = cell_area

cell_SA_w[cell_SA_w == 0] = 1
cell_SA_c[cell_SA_c == 0] = 1

coral_frac_w = coral_grid_w/cell_SA_w
coral_frac_c = coral_grid_c/cell_SA_c

# Remove coral frac > 1 (note that this only affects 0.1% of coral cells, and
# most are within 10%. Coral frac > 1 can appear due to the shifting of corals)
coral_frac_w[coral_frac_w > 1] = 1
coral_frac_c[coral_frac_c > 1] = 1

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

# See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

# Parameters
av_cells_per_cluster = param['av_cells_per_cluster']

# Create a dict to store the cell id for each EEZ
cell_id_dict = {}

# Get a list of all ISO codes
iso_list = np.sort(np.unique(eez_grid_w[eez_grid_w != 0]))
iso_list[[0, np.where(iso_list == 690)[0][0]]] = iso_list[[np.where(iso_list == 690)[0][0], 0]]

# Define certain model-specific parameters for different regions
dist_thresh_dict = {86 : 1.5,
                    174: 1.5,
                    250: 1.5,
                    404: 1.0,
                    450: 2.5,
                    462: 1.0,
                    480: 2.0,
                    508: 2.0,
                    690: 0.3,
                    706: 1.5,
                    834: 0.7
                    }

print('Grouping coral cells into clusters...')
for i, iso_code in tqdm(enumerate(iso_list), total=len(iso_list)):
    # 1. Carry out agglomerative clustering (X, model)
    lon_w, lat_w = np.meshgrid(lon_rho_w, lat_rho_w)
    lon_w = lon_w[eez_grid_w == iso_code]
    lat_w = lat_w[eez_grid_w == iso_code]
    n_clusters = int(len(lon_w)/av_cells_per_cluster)

    X = np.concatenate((lon_w, lat_w)).reshape((2,-1)).T

    # Use agglomerative clustering with a distance threshold
    knn_graph = kneighbors_graph(X, n_clusters, include_self=False)
    linkage='ward'
    model = AgglomerativeClustering(linkage=linkage, connectivity=knn_graph,
                                    n_clusters=None,
                                    distance_threshold=dist_thresh_dict[iso_code]).fit(X)

    X = np.concatenate([X, model.labels_.reshape(-1, 1)], axis=1)

    # 2. For Seychelles only: manually relabel some sites as per April's sites
    if iso_code == 690:

        assert i == 0

        def relabel_island(X, lon_w, lon_e, lat_s, lat_n, **kwargs):
            label_list = np.unique(X[:, 2][np.where((X[:, 0] < lon_e)*(X[:, 0] > lon_w)*(X[:, 1] > lat_s)*(X[:, 1] < lat_n))])

            if 'label_pos' in kwargs:
                X[:, 2][np.where((X[:, 0] < lon_e)*(X[:, 0] > lon_w)*(X[:, 1] > lat_s)*(X[:, 1] < lat_n))] = label_list[kwargs['label_pos']]
            else:
                X[:, 2][np.where((X[:, 0] < lon_e)*(X[:, 0] > lon_w)*(X[:, 1] > lat_s)*(X[:, 1] < lat_n))] = label_list[0]

            return X

        # Get the labels for Aldabra
        aldabra_labels = np.unique(X[:, 2][np.where((X[:, 0] < 47)*(X[:, 1] > -9.6)*(X[:, 1] < -9.0))])

        # Relabel
        X[:, 2][np.where((X[:, 0] < 46.35)*(X[:, 0] > 46.00)*(X[:, 1] < -9.25)*(X[:, 1] > -9.42))] = aldabra_labels[0]
        X[:, 2][np.where((X[:, 0] < 46.35)*(X[:, 0] > 46.00)*(X[:, 1] <= -9.42)*(X[:, 1] > -9.60))] = aldabra_labels[1]
        X[:, 2][np.where((X[:, 0] < 46.70)*(X[:, 0] >= 46.35)*(X[:, 1] < -9.25)*(X[:, 1] > -9.42))] = aldabra_labels[2]
        X[:, 2][np.where((X[:, 0] < 46.70)*(X[:, 0] >= 46.35)*(X[:, 1] <= -9.42)*(X[:, 1] > -9.60))] = aldabra_labels[3]

        # Relabel Cosmoledo
        X = relabel_island(X, 47.2, 47.75, -9.9, -9.5)

        # Relabel Farquhar
        X = relabel_island(X, 50.8, 51.4, -10.4, -10.0)

        # Relabel Providence
        X = relabel_island(X, 50.85, 51.2, -9.7, -9.1)

        # Correct Alphonse Group
        X = relabel_island(X, 52.5, 52.9, -7.3, -7.06, label_pos=-1)

        # Relabel Desroches
        X = relabel_island(X, 53.5, 53.8, -5.8, -5.4)

        # Relabel Platte
        X = relabel_island(X, 55.1, 55.5, -6.1, -5.6)

        # Relabel Coetivy
        X = relabel_island(X, 56, 56.6, -7.5, -6.9)

    # 3. Relabel in latitude order
    #    Firstly find the midpoint latitude for each cluster
    old_labels = np.unique(X[:, 2])
    mean_latitude = np.zeros_like(old_labels)
    new_order = np.zeros_like(old_labels)
    new_labels = np.zeros_like(X[:, 2])

    for j in range(len(old_labels)):
        mean_latitude[j] = np.mean(X[:, 1][np.where(X[:, 2] == old_labels[j])])

    # Make sure all average latitudes are unique to avoid merging clusters
    assert len(np.unique(mean_latitude)) == len(old_labels)

    new_order = old_labels[mean_latitude.argsort()]

    for j in range(len(old_labels)):
        new_labels[np.where(X[:, 2] == new_order[j])] = int(j)

    X[:, 2] = new_labels + 1

    # 4. Save to X_w
    if i == 0:
        cell_id_dict[str(iso_code)] = np.unique(X[:, 2]).compressed()
        X_w = np.copy(X)
    else:
        X[:, 2] += np.max(X_w[:, 2])
        assert ~np.isin(np.unique(X[:, 2]), np.unique(X_w[:, 2])).any()
        cell_id_dict[str(iso_code)] = np.unique(X[:, 2]).compressed()
        X_w = np.concatenate([X_w, X], axis=0)

# Now form an equivalent array for CMEMS
lon_c, lat_c = np.meshgrid(lon_psi_c, lat_psi_c)
lon_c = lon_c[eez_grid_c > 0]
lat_c = lat_c[eez_grid_c > 0]

X_c = np.concatenate([lon_c, lat_c, np.zeros_like(lon_c)]).reshape((3, -1)).T

# Find the nearest point in X_w for each point in X_c and assign that label
nn_dist, nn_nearest_pt = KDTree(X_w[:, :2]).query(X_c[:, :2], k=1, p=2)
assert np.max(nn_dist) < 0.2
X_c[:, 2] = X_w[:, 2][nn_nearest_pt]

# Now grid both
coral_grp_c = np.zeros_like(coral_grid_c)
coral_grp_w = np.zeros_like(coral_grid_w)

lon_c, lat_c = np.meshgrid(lon_psi_c, lat_psi_c)
lon_w, lat_w = np.meshgrid(lon_rho_w, lat_rho_w)

print('Gridding coral clusters (part 1)...')
for i in tqdm(range(np.shape(X_c)[0]), total=np.shape(X_c)[0]):
    lon_i = X_c[i, 0]
    lat_i = X_c[i, 1]
    grp_i = X_c[i, 2]

    coral_grp_c[np.where((lon_c == lon_i)*(lat_c == lat_i))] = grp_i

print('Gridding coral clusters (part 2)...')
for i in tqdm(range(np.shape(X_w)[0]), total=np.shape(X_w)[0]):
    lon_i = X_w[i, 0]
    lat_i = X_w[i, 1]
    grp_i = X_w[i, 2]

    coral_grp_w[np.where((lon_w == lon_i)*(lat_w == lat_i))] = grp_i

# Check that no groups have been lost
assert np.array_equiv(np.unique(coral_grp_c), np.unique(coral_grp_w))

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

###############################################################################
# Export to netcdf ############################################################
###############################################################################

with Dataset(fh['out'], mode='w') as nc:
    # Create the dimensions
    nc.createDimension('lon_rho_w', len(lon_rho_w))
    nc.createDimension('lat_rho_w', len(lat_rho_w))
    nc.createDimension('lon_psi_w', len(lon_psi_w))
    nc.createDimension('lat_psi_w', len(lat_psi_w))

    nc.createDimension('lon_rho_c', len(lon_rho_c))
    nc.createDimension('lat_rho_c', len(lat_rho_c))
    nc.createDimension('lon_psi_c', len(lon_psi_c))
    nc.createDimension('lat_psi_c', len(lat_psi_c))

    # Create the variables (axes)
    nc.createVariable('lon_rho_w', 'f4', ('lon_rho_w'), zlib=True)
    nc.variables['lon_rho_w'].long_name = 'longitude_rho_winds'
    nc.variables['lon_rho_w'].units = 'degrees_east'
    nc.variables['lon_rho_w'].standard_name = 'longitude_on_rho_points_for_winds'
    nc.variables['lon_rho_w'][:] = lon_rho_w

    nc.createVariable('lat_rho_w', 'f4', ('lat_rho_w'), zlib=True)
    nc.variables['lat_rho_w'].long_name = 'latitude_rho_winds'
    nc.variables['lat_rho_w'].units = 'degrees_north'
    nc.variables['lat_rho_w'].standard_name = 'latitude_on_rho_points_for_winds'
    nc.variables['lat_rho_w'][:] = lat_rho_w

    nc.createVariable('lon_psi_w', 'f4', ('lon_psi_w'), zlib=True)
    nc.variables['lon_psi_w'].long_name = 'longitude_psi_winds'
    nc.variables['lon_psi_w'].units = 'degrees_east'
    nc.variables['lon_psi_w'].standard_name = 'longitude_on_psi_points_for_winds'
    nc.variables['lon_psi_w'][:] = lon_psi_w

    nc.createVariable('lat_psi_w', 'f4', ('lat_psi_w'), zlib=True)
    nc.variables['lat_psi_w'].long_name = 'latitude_psi_winds'
    nc.variables['lat_psi_w'].units = 'degrees_north'
    nc.variables['lat_psi_w'].standard_name = 'latitude_on_psi_points_for_winds'
    nc.variables['lat_psi_w'][:] = lat_psi_w

    nc.createVariable('lon_rho_c', 'f4', ('lon_rho_c'), zlib=True)
    nc.variables['lon_rho_c'].long_name = 'longitude_rho_cmems'
    nc.variables['lon_rho_c'].units = 'degrees_east'
    nc.variables['lon_rho_c'].standard_name = 'longitude_on_rho_points_for_cmems'
    nc.variables['lon_rho_c'][:] = lon_rho_c

    nc.createVariable('lat_rho_c', 'f4', ('lat_rho_c'), zlib=True)
    nc.variables['lat_rho_c'].long_name = 'latitude_rho_cmems'
    nc.variables['lat_rho_c'].units = 'degrees_north'
    nc.variables['lat_rho_c'].standard_name = 'latitude_on_rho_points_for_cmems'
    nc.variables['lat_rho_c'][:] = lat_rho_c

    nc.createVariable('lon_psi_c', 'f4', ('lon_psi_c'), zlib=True)
    nc.variables['lon_psi_c'].long_name = 'longitude_psi_cmems'
    nc.variables['lon_psi_c'].units = 'degrees_east'
    nc.variables['lon_psi_c'].standard_name = 'longitude_on_psi_points_for_cmems'
    nc.variables['lon_psi_c'][:] = lon_psi_c

    nc.createVariable('lat_psi_c', 'f4', ('lat_psi_c'), zlib=True)
    nc.variables['lat_psi_c'].long_name = 'latitude_psi_cmems'
    nc.variables['lat_psi_c'].units = 'degrees_north'
    nc.variables['lat_psi_c'].standard_name = 'latitude_on_psi_points_for_cmems'
    nc.variables['lat_psi_c'][:] = lat_psi_c

    # Create the variables (fields)
    nc.createVariable('lsm_w', 'i2', ('lat_rho_w', 'lon_rho_w'), zlib=True)
    nc.variables['lsm_w'].long_name = 'land_sea_mask_on_rho_grid_winds'
    nc.variables['lsm_w'].standard_name = 'land_sea_mask_on_rho_grid_for_winds'
    nc.variables['lsm_w'][:] = lsm_rho_w

    nc.createVariable('lsm_c', 'i2', ('lat_psi_c', 'lon_psi_c'), zlib=True)
    nc.variables['lsm_c'].long_name = 'land_sea_mask_on_psi_grid_cmems'
    nc.variables['lsm_c'].standard_name = 'land_sea_mask_on_psi_grid_for_cmems'
    nc.variables['lsm_c'][:] = lsm_psi_c

    # Set fill values
    fill_value = -999
    def set_fill(grid, old_fill, new_fill):
        grid[grid == old_fill] = new_fill
        return grid

    nc.createVariable('eez_w', 'i2', ('lat_rho_w', 'lon_rho_w'), zlib=True, fill_value=fill_value)
    nc.variables['eez_w'].long_name = 'EEZ_on_rho_grid_winds'
    nc.variables['eez_w'].standard_name = 'EEZ_on_rho_grid_for_winds'
    nc.variables['eez_w'][:] = set_fill(eez_grid_w, 0, fill_value)

    nc.createVariable('eez_c', 'i2', ('lat_psi_c', 'lon_psi_c'), zlib=True, fill_value=fill_value)
    nc.variables['eez_c'].long_name = 'EEZ_on_psi_grid_cmems'
    nc.variables['eez_c'].standard_name = 'EEZ_on_psi_grid_for_cmems'
    nc.variables['eez_c'][:] = set_fill(eez_grid_c, 0, fill_value)

    nc.createVariable('coral_cover_w', 'i4', ('lat_rho_w', 'lon_rho_w'), zlib=True, fill_value=fill_value)
    nc.variables['coral_cover_w'].long_name = 'coral_cover_on_rho_grid_winds'
    nc.variables['coral_cover_w'].standard_name = 'coral_cover_on_rho_grid_for_winds'
    nc.variables['coral_cover_w'][:] = set_fill(coral_grid_w, 0, fill_value)

    nc.createVariable('coral_cover_c', 'i4', ('lat_psi_c', 'lon_psi_c'), zlib=True, fill_value=fill_value)
    nc.variables['coral_cover_c'].long_name = 'coral_cover_on_psi_grid_cmems'
    nc.variables['coral_cover_c'].standard_name = 'coral_cover_on_psi_grid_for_cmems'
    nc.variables['coral_cover_c'][:] = set_fill(coral_grid_c, 0, fill_value)

    nc.createVariable('coral_frac_w', 'f4', ('lat_rho_w', 'lon_rho_w'), zlib=True, fill_value=fill_value)
    nc.variables['coral_frac_w'].long_name = 'coral_fraction_on_rho_grid_winds'
    nc.variables['coral_frac_w'].standard_name = 'coral_fraction_on_rho_grid_for_winds'
    nc.variables['coral_frac_w'][:] = set_fill(coral_frac_w, 0, fill_value)

    nc.createVariable('coral_frac_c', 'f4', ('lat_psi_c', 'lon_psi_c'), zlib=True, fill_value=fill_value)
    nc.variables['coral_frac_c'].long_name = 'coral_fraction_on_psi_grid_cmems'
    nc.variables['coral_frac_c'].standard_name = 'coral_fraction_on_psi_grid_for_cmems'
    nc.variables['coral_frac_c'][:] = set_fill(coral_frac_c, 0, fill_value)

    nc.createVariable('coral_grp_w', 'i2', ('lat_rho_w', 'lon_rho_w'), zlib=True, fill_value=fill_value)
    nc.variables['coral_grp_w'].long_name = 'coral_group_on_rho_grid_winds'
    nc.variables['coral_grp_w'].standard_name = 'coral_group_on_rho_grid_for_winds'
    nc.variables['coral_grp_w'][:] = set_fill(coral_grp_w, 0, fill_value)

    nc.createVariable('coral_grp_c', 'i2', ('lat_psi_c', 'lon_psi_c'), zlib=True, fill_value=fill_value)
    nc.variables['coral_grp_c'].long_name = 'coral_group_on_psi_grid_cmems'
    nc.variables['coral_grp_c'].standard_name = 'coral_group_on_psi_grid_for_cmems'
    nc.variables['coral_grp_c'][:] = set_fill(coral_grp_c, 0, fill_value)
